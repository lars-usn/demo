"""Calculate the wavefronts from a line array.

The array consists of N elements described as points.
Wavefronts are plotted as spherical waves from each element

An interactive version can be run from the Jupyter Notebook
'wavefront_demo.ipynb'
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
from pathlib import Path
import ipywidgets

COLOR = {
    "transducer": "#A63D1F",  # "#B64926"  "#A63D1F" "#B35A1F" "#8C2D19"
    "transducer_background": "#F0FBFF",  # "#D6EFFC "#C2E7F7" "#E0F4FC"
    "text_face": "#F0FBFF",  # "#E6F3F7", " # "#F0FBFF", "#EAF7FA"
    "text_edge": "#7AA6B8",
    "background": "#EAF6FF",  # DDF2FF, E6F7FA F0FBFF
}

LINEFORMAT = {
    "wavefront": {
        "color": "#FF6B35",  # FF6B35 FF5E3A FF4500 FF7A00 FF7043
        "linestyle": "dotted",
        "linewidth": 1,
    },
    "indicator": {
        "color": "#1F77B4",
        "linestyle": "dotted",
        "linewidth": 1,
    },
    "reference": {"color": "C0", "linestyle": "dashed"},
    "source": {
        "color": COLOR["transducer"],
        "linestyle": "none",
        "marker": "s",
        "markersize": 8,
    },
}

LOGOFILE = "usn-logo-purple.png"
FIGURE_NAME = "Transducer Array Beamprofile"


class ArrayWaves:
    def __init__(self, create_widgets=False):
        self.n_elements = 6
        self.c = 1500
        self.frequency = 100e3
        self.steering_angle = 0

        # Timing
        self.time_step = 0.1
        self.duration = 10.0
        self.animation = None

        # Display
        self.pitch = 1.0  # Large pitch for demonstration
        self.x_lim = 6.0
        self.z_max = 24.0
        self.x_ax_range = 6.0 * np.array([-1, 1])
        self.n_points = 100
        self._radius_steps = None

        self.theta_wavefront = np.radians(np.linspace(-90, 90, 50))
        self.cos_wavefront = np.cos(self.theta_wavefront)
        self.sin_wavefront = np.sin(self.theta_wavefront)

        # Initialise figures and values
        self.fig, self.axes, self.graphs = self._initialise_graphs()
        self.draw_wavefield_axis()
        self.draw_sources()
        self.update_resulttext()

        if create_widgets:
            self.widget_layout, self.widgets = self._create_widgets()

    @property
    def wavelength(self):
        """Calculate acoustic wavelength."""
        return self.c / self.frequency

    @property
    def delay_diff(self):
        """Find delay between elements."""
        return self.pitch * np.sin(self.steering_angle) / self.c

    @property
    def delay(self):
        """Find delay vector."""
        delay = self.delay_diff * np.arange(self.n_elements)
        delay -= delay.min()

        return delay

    @property
    def n_time_steps(self):
        return int(round(self.duration / self.time_step))

    @property
    def radius_time_steps(self):
        return np.linspace(0, self.z_max, self.n_time_steps)

    @property
    def x_sources(self):
        """Elements positions along x-axis, z=0."""
        rel_positions = np.arange(self.n_elements) - (self.n_elements - 1) / 2

        return self.pitch * rel_positions

    def calculate_wavefront(self, x0, radius):
        """
        Calculate wavefront as a circular arc around an element.
        x is lateral dimension (y-axis).
        z is axial dimension (depth) (x-axis).

        Parameters
        ----------
        x0 : float
            Lateral position [m]
        radius : float
            Radius of wavefront

        Returns
        -------
        z : ndarray
            Axial coordinates

        x : ndarray
            Lateral coordinates
        """
        if radius >= 0:
            z = radius * self.cos_wavefront
            x = radius * self.sin_wavefront
        else:
            z = [0]
            x = [0]

        return z, x + x0

    def draw_wavefronts(self, x_sources, radii):
        """Draw wavefronts around sources in lateral xy-plane."""

        for x0, radius, graph in zip(
            x_sources,
            radii,
            self.graphs["wavefronts"],
        ):
            z, x = self.calculate_wavefront(x0, radius)
            graph.set_data(x, z)

    def display_wavefront(self, r0=0):
        """Update wavefronts for a specified propagation distance."""
        radii = r0 - self.c * self.delay

        self.draw_wavefronts(self.x_sources, radii)

    def _animation_frame(self, frame):
        """Update artists for a single animation frame."""
        self.display_wavefront(self._radius_steps[frame])
        return self.graphs["wavefronts"]

    def run_animation(self):
        """Run wavefield animation."""
        self.stop_animation()
        self.draw_wavefield_axis()

        self._radius_steps = self.radius_time_steps
        n_steps = len(self._radius_steps)

        self.animation = FuncAnimation(
            self.fig,
            self._animation_frame,
            frames=n_steps,
            interval=int(1000 * self.time_step),
            repeat=True,
        )
        self.fig.canvas.draw_idle()

    def stop_animation(self):
        if (
            self.animation is not None
            and self.animation.event_source is not None
        ):
            self.animation.event_source.stop()

        self.animation = None

    def draw_wavefield_axis(self):
        """Draw orientation axis and empty wavefront lines."""

        # Indicator for direction
        z_max_disp = self.z_max
        self.graphs["direction"].set_data(
            [0, z_max_disp * np.cos(self.steering_angle)],
            [0, z_max_disp * np.sin(self.steering_angle)],
        )

        # Empty wavefront lines
        for graph in self.graphs["wavefronts"]:
            graph.set_data([], [])

    def draw_sources(self):
        """
        Mark source positions and create new set of wavefronts.

        Wavefront lines are recreated because the number of array elements
        may change during operation.
        """
        # Draw sources
        self.graphs["sources"].set_data(
            np.zeros_like(self.x_sources),
            self.x_sources,
        )

        # Create new empty wavefront graphs
        for graph in self.graphs["wavefronts"]:
            graph.remove()

        self.graphs["wavefronts"] = [
            self.axes["wavefronts"].plot([], [], **LINEFORMAT["wavefront"])[0]
            for _ in range(self.n_elements)
        ]

    def update_resulttext(self):
        """Update text box with array parameters."""

        value_lines = [
            f"{self.n_elements}",
            rf"{np.degrees(self.steering_angle):.0f}$^\circ$",
        ]

        for line_no, value in enumerate(value_lines):
            self.graphs["text"][(line_no, 2)].get_text().set_text(value)

    def _initialise_graphs(self):
        """Initialise result graphs."""
        plt.close(FIGURE_NAME)

        wavefront_row = ["wavefronts"] * 6

        fig, axes = plt.subplot_mosaic(
            [
                ["."] + wavefront_row,
                ["text"] + wavefront_row,
                ["text"] + wavefront_row,
                ["."] + wavefront_row,
                ["."] + wavefront_row,
                ["logo"] + wavefront_row,
            ],
            figsize=(18, 6),
            layout="constrained",
            num=FIGURE_NAME,
        )

        graphs = self._create_wavefront_plot(axes["wavefronts"])
        graphs["text"] = self._create_resulttextbox(axes["text"])
        self._create_logo(axes["logo"])

        return fig, axes, graphs

    def _create_wavefront_plot(self, ax):
        """
        Create axis for beam profile graphs.

        Parameters
        ----------
        ax : Axis object
            Axis where wavefronts are plotted
        """

        # Axes
        ax.set(
            xlabel="Distance",
            ylabel="Transducer Array",
            title="Wavefront image",
            aspect="equal",
            xlim=(0, self.z_max),
            ylim=self.x_ax_range,
        )

        # Remove axes, but keep coloured background
        ax.grid(visible=False)
        ax.set_facecolor(COLOR["background"])

        ax.tick_params(
            left=False,
            bottom=False,
            labelleft=False,
            labelbottom=False,
        )

        for spine in ax.spines.values():
            spine.set_visible(False)

        # Create artists for plots
        graphs = {}

        # Wavefront image
        graphs["wavefronts"] = [
            ax.plot([], [], **LINEFORMAT["wavefront"])[0]
            for _ in range(self.n_elements)
        ]

        # Acoustic axis, normally not changed
        graphs["acoustic_axis"] = ax.plot(
            [0, self.z_max],
            [0, 0],
            **LINEFORMAT["indicator"],
        )[0]

        # Sources, steering direction, changed during runtime
        graphs["sources"] = ax.plot([], [], **LINEFORMAT["source"])[0]
        graphs["direction"] = ax.plot([], [], **LINEFORMAT["indicator"])[0]

        return graphs

    def _create_resulttextbox(self, ax):
        """
        Create and attach a formatted results text box to an Axes.

        The text box is anchored to an axis and remains fixed relative to
        the axes if the figure is resized.

        Parameters
        ----------
        ax : Axis object
            Axis where text is shown

        Returns
        -------
        matplotlib.table.Table
            Handle to results table.
        """
        ax.axis("off")

        resulttext = [
            ["No. of elements", "$N_{el}$", "-"],
            ["Steering angle", r"$\theta_s$", "-"],
        ]

        table = ax.table(
            cellText=resulttext,
            loc="upper left",
            cellLoc="left",
            colWidths=[0.50, 0.25, 0.25],
        )

        for cell in table.get_celld().values():
            cell.set_linewidth(0.2)
            cell.visible_edges = "TB"
            cell.set_facecolor(COLOR["text_face"])
            cell.PAD = 0.03
            cell.set_text_props(fontfamily="DejaVu Sans")

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.1)

        for r in range(len(resulttext)):
            table[(r, 1)].set_text_props(ha="center")
            table[(r, 2)].set_text_props(ha="left")

        return table

    def _create_logo(self, ax):
        """
        Load logo file and place in specified axis.

        Parameters
        ----------
        ax : Axis object
            Axis where logo image is shown
        """
        ax.set_axis_off()

        try:
            base_path = Path(__file__).resolve().parent
        except NameError:
            # Running in Jupyter
            base_path = Path.cwd()

        logo_path = (base_path / ".." / "figs" / LOGOFILE).resolve()

        if logo_path.exists():
            img = mpimg.imread(logo_path)
            ax.imshow(img)
        else:
            ax.text(
                0.5,
                0.5,
                "USN",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    def _refresh(self):
        self.fig.canvas.draw_idle()

    # === Widget callbacks =========================================
    def _run_animation_callback(self, button):
        self.run_animation()

    def _stop_animation_callback(self, button):
        self.stop_animation()

    def _radius_change_callback(self, change):
        self.stop_animation()
        self.display_wavefront(change["new"])
        self._refresh()

    def _steering_change_callback(self, change):
        self.stop_animation()
        self.steering_angle = np.radians(change["new"])
        self.draw_wavefield_axis()
        self.update_resulttext()

        radius = self.widgets["radius"].value
        self.display_wavefront(radius)
        self._refresh()

    # === Interactive widgets ======================================
    def _create_widgets(self):
        """Create widgets for interactive operation."""
        title = "Wavefronts from Transducer Array"
        title_widget = ipywidgets.Label(
            title,
            style=dict(font_weight="bold"),
        )

        slider_layout = {
            "continuous_update": True,
            "layout": ipywidgets.Layout(width="95%"),
            "style": {"description_width": "30%"},
        }

        text_width = "20%"
        slider_width = "95%"

        # === Define widgets
        radius_widget = ipywidgets.FloatSlider(
            value=0,
            min=0,
            max=self.z_max,
            step=0.1,
            description="Distance [m]",
            **slider_layout,
        )
        radius_widget.observe(
            self._radius_change_callback,
            names="value",
        )

        steering_angle_widget = ipywidgets.FloatSlider(
            min=-90,
            max=90,
            value=0,
            step=1,
            readout_format=".0f",
            description="Steering angle [Deg.]",
            **slider_layout,
        )
        steering_angle_widget.observe(
            self._steering_change_callback,
            names="value",
        )

        animate_widget = ipywidgets.Button(
            description="Run",
            button_style="primary",
            icon="play",
        )
        animate_widget.on_click(self._run_animation_callback)

        stop_widget = ipywidgets.Button(
            description="Stop",
            button_style="danger",
            icon="stop",
        )
        stop_widget.on_click(self._stop_animation_callback)

        # === Widget layout
        array_parameter_column = ipywidgets.VBox(
            [
                animate_widget,
                stop_widget,
            ],
            layout=ipywidgets.Layout(width=text_width),
        )

        slider_column = ipywidgets.VBox(
            [radius_widget, steering_angle_widget],
            layout=ipywidgets.Layout(width=slider_width),
        )

        widget_layout = ipywidgets.HBox(
            [
                array_parameter_column,
                slider_column,
            ],
            layout=ipywidgets.Layout(width="80%"),
        )

        widget_layout = ipywidgets.VBox([title_widget, widget_layout])

        widget = {
            "radius": radius_widget,
            "steering_angle": steering_angle_widget,
            "animate": animate_widget,
        }

        return widget_layout, widget
