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
        """Colors associated with each direction."""
        color_map = {
            Direction.INCOMING: "#000080",
            Direction.REFLECTED: "#800000",
            Direction.TRANSMITTED: "#004C00",
        }
        return color_map[self]

    @property
    def angle_radius(self):
        """Radius of angle arc, avoid overlap."""
        radius_map = {
            Direction.INCOMING: 0.25,
            Direction.REFLECTED: 0.20,
            Direction.TRANSMITTED: 0.25,
        }
        return radius_map[self]

    def start_end(self, x, y):
        """Start end end coordinates of arrows, depending on direction."""
        if self is Direction.INCOMING:
            return (x, y), (0, 0)

        elif self is Direction.REFLECTED:
            return (0, 0), (-x, y)

        elif self is Direction.TRANSMITTED:
            return (0, 0), (-x, -y)

        raise ValueError(f"Unknown direction: {self}")

    def arc_angles(self, angle):
        """Define start and end of angle arc, depending on direction."""
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
        self.z = [1.5, 1.6]

        self.fig, self.axes = self.initialise_graphs()

        self.angle_artists = []  # All artists with angles: Arcs, labels
        self.parameter_labels = []  # Labels on axis: Sound speeds
        self.widget_layout = None
        self.widgets = None

        self.update_arrows()

        if create_widgets:
            self.widget_layout = self._create_widgets()

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
    def reflection(self):
        """Pressure reflection coefficienbt."""
        if self.angle_transmitted is None:
            return 1.0

        z = self.z
        cos_i = np.cos(np.radians(self.angle_in))
        cos_t = np.cos(np.radians(self.angle_transmitted))

        return (z[1] * cos_i - z[0] * cos_t) / (z[1] * cos_i + z[0] * cos_t)

    @property
    def transmission(self):
        """Pressure transmission coefficient."""
        if self.angle_transmitted is None:
            return 0.0

        z = self.z
        cos_i = np.cos(np.radians(self.angle_in))
        cos_t = np.cos(np.radians(self.angle_transmitted))

        return (2 * z[1] * cos_i) / (z[1] * cos_i + z[0] * cos_t)

    @property
    def reflection_power(self):
        """Power reflection coefficient."""
        return self.reflection**2

    @property
    def transmission_power(self):
        """Power transmission coefficient."""
        return 1 - self.reflection_power

    @property
    def angles(self):
        return (self.angle_in, self.angle_in, self.angle_transmitted)

    @property
    def directions(self):
        return (Direction.INCOMING, Direction.REFLECTED, Direction.TRANSMITTED)

    @property
    def lengths(self):
        lengths = np.array(
            [1.0, self.reflection_power, self.transmission_power]
        )
        lengths = np.clip(lengths, 0.10, 1.0)
        return lengths

    def arrow_start_end(self, angle, direction, length):
        """
        Find start and end positions of arrows.

        Parameters
        ----------
        angle : float
            Arrow angle in degrees
        direction : enum Direction
            Incoming, reflected or transmitted wave
        length : float
            Relative length of arrow

        Returns
        -------
        start : tuple (x,y)
            Start coordinates of arrow
        end : tuple (x,y)
            End coordinates of arrow
        """

        length = self.arrow_length * length
        phi = np.radians(angle)
        x = -length * np.sin(phi)
        y = length * np.cos(phi)

        return direction.start_end(x, y)

    def update_arrows(self):
        """Update arrows, angle markers, and annotations."""

        self.remove_artists(self.angle_artists)
        self.remove_artists(self.parameter_labels)

        for arrow, angle, direction, length in zip(
            self.arrows, self.angles, self.directions, self.lengths
        ):

            if angle is None:
                arrow.set_visible(False)
            else:
                arrow.set_visible(True)
                start, end = self.arrow_start_end(angle, direction, length)
                arrow.set_positions(start, end)

                if direction != Direction.REFLECTED:
                    self.draw_angle(angle, direction)

        self.annotate_graph()
        self.fig.canvas.draw_idle()

    def annotate_graph(self):
        """Write parameter values on the axes."""

        ax = self.axes["drawing"]

        for k, (c, z, y) in enumerate(zip(self.c, self.z, (0.65, 0.35))):
            label = ax.text(
                0.05,
                y,
                (
                    rf"$c_{k+1} = {c:.0f}\,\mathrm{{m/s}}$"
                    "\n"
                    rf"$z_{k+1} = {z:.1f}\,\mathrm{{MRayl}}$"
                ),
                transform=ax.transAxes,
                ha="left",
                va="center",
            )
            self.parameter_labels.append(label)

        rt_values = [self.reflection_power, self.transmission_power]
        rt_text = ["Reflected", "Transmitted"]

        for rt, rt_text, y in zip(rt_values, rt_text, (0.97, 0.03)):
            rt_label = ax.text(
                0.51,
                y,
                rt_text + rf" power: {rt:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="center",
            )
            self.parameter_labels.append(rt_label)

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

        text = "Critical angle\n" r"$\sin\theta_c = c_1/c_2$" "\n"

        if self.c[1] > self.c[0]:
            theta_crit = np.degrees(np.arcsin(self.c[0] / self.c[1]))

            text += rf"$\theta_c = {theta_crit:.1f}^\circ$"
        else:
            text += "No critical angle"

        if self.angle_in > 0:
            c2_critical = self.c[0] / np.sin(np.radians(self.angle_in))

            text += (
                "\n\n"
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
                start, end = self.arrow_start_end(angle, direction, 1.0)
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

    def _impedance_change_callback(self, change):
        index = change["owner"].medium_index
        self.z[index] = float(change["new"])
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

        panel_layout = widgets.Layout(width="95%", max_width="1000px")
        column_slider_layout = widgets.Layout(width="90%")
        angle_slider_layout = widgets.Layout(width="98%")

        title_widget = widgets.HTML(
            value="<h3>Illustration of Snell's Law</h3>",
            layout=widgets.Layout(width="100%", margin="0 0 10px 0"),
        )

        angle_widget = widgets.FloatSlider(
            value=self.angle_in,
            min=0,
            max=90,
            step=1.0,
            readout_format=".0f",
            description="Angle of incoming wave [°]",
            continuous_update=True,
            layout=angle_slider_layout,
            style={"description_width": "160px"},
        )

        angle_widget.observe(
            self._angle_change_callback,
            names="value",
        )

        soundspeed_title_widget = widgets.HTML(
            value="Speed of sound [m/s]",
            layout=widgets.Layout(
                width="100%",
                margin="0 0 5px 0",
            ),
        )

        impedance_title_widget = widgets.HTML(
            value="Characteristic acoustic impedance [MRayl]",
            layout=widgets.Layout(
                width="100%",
                margin="0 0 5px 0",
            ),
        )

        medium_widgets = []
        soundspeed_widgets = []
        impedance_widgets = []

        slider_style = {"description_width": "0px"}

        for k in range(2):
            medium_widget = widgets.HTML(value=f"Medium {k+1}")
            medium_widgets.append(medium_widget)

            speed_widget = widgets.FloatSlider(
                value=self.c[k],
                min=500,
                max=3000,
                step=10,
                readout_format=".0f",
                description="",
                continuous_update=True,
                layout=column_slider_layout,
                style=slider_style,
            )

            speed_widget.medium_index = k
            speed_widget.observe(
                self._sound_speed_change_callback,
                names="value",
            )

            soundspeed_widgets.append(speed_widget)

            impedance_widget = widgets.FloatSlider(
                value=self.z[k],
                min=1.0,
                max=4.0,
                step=0.1,
                readout_format=".1f",
                description=f"",
                continuous_update=True,
                layout=column_slider_layout,
                style=slider_style,
            )
            impedance_widget.medium_index = k
            impedance_widget.observe(
                self._impedance_change_callback,
                names="value",
            )

            impedance_widgets.append(impedance_widget)

        parameter_layout = widgets.GridBox(
            children=[
                widgets.HTML(value=""),
                widgets.HTML(value="Speed of sound [m/s]"),
                widgets.HTML(
                    value="Characteristic acoustic impedance [MRayl]"
                ),
                medium_widgets[0],
                soundspeed_widgets[0],
                impedance_widgets[0],
                medium_widgets[1],
                soundspeed_widgets[1],
                impedance_widgets[1],
            ],
            layout=widgets.Layout(
                width="100%",
                grid_template_columns="100px minmax(300px, 1fr) minmax(300px, 1fr)",
                grid_template_rows="auto auto auto",
                grid_gap="10px 25px",
                align_items="center",
            ),
        )

        widget_layout = widgets.VBox(
            [
                title_widget,
                parameter_layout,
                angle_widget,
            ],
            layout=panel_layout,
        )

        return widget_layout


# =============================================================================
#         medium_column = widgets.VBox(
#             [soundspeed_title_widget, *medium_widgets]
#         )
#
#         soundspeed_column = widgets.VBox(
#             [soundspeed_title_widget, *soundspeed_widgets]
#         )
#
#         impedance_column = widgets.VBox(
#             [
#                 impedance_title_widget,
#                 *impedance_widgets,
#             ]
#         )
#
#         parameter_layout = widgets.HBox(
#             [
#                 medium_column,
#                 soundspeed_column,
#                 impedance_column,
#             ],
#             layout=widgets.Layout(
#                 width="100%",
#                 display="flex",
#                 gap="40px",
#                 align_items="flex-start",
#             ),
#         )
#
#         widget_layout = widgets.VBox(
#             [
#                 title_widget,
#                 parameter_layout,
#                 angle_widget,
#             ]
#         )
#
#         return widget_layout
#
# =============================================================================
