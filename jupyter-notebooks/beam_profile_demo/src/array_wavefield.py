"""Calculate the wavefronts from a line array.

The array consists of N elements described as points.
Wavefronts are plotted as sperical waves rom each element

An interactive version can be run from the Jupyter Notebook
'wavefront_demo.ipynb'
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import time

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
        "linestyle": "dashed",
        "linewidth": 4,
    },
    "indicator": {"color": "#1F77B4", "linestyle": "dotted"},
    "reference": {"color": "C0", "linestyle": "dashed"},
    "source": {
        "color": COLOR["transducer"],
        "linestyle": "none",
        "marker": "s",
        "markersize": 30,
    },
}

LOGOFILE = "usn-logo-purple.png"
FIGURE_NAME = "Transducer Array Beamprofile"


class ArrayWaves:
    def __init__(self):
        self.n_elements = 4
        self.c = 1500
        self.frequency = 100e3
        self.steering_angle = 0

        # Timing
        self.time_step = 0.1
        self.duration = 10.0

        # Display
        self.x_lim = 4.0
        self.z_max = 12.0
        self.x_ax_range = 6.0 * np.array([-1, 1])
        self.n_points = 100

        # Initialise figures and values
        self.fig, self.axes, self.graphs = self._initialise_graphs()
        self.update_wavefield_graphs()

    @property
    def wavelength(self):
        """Calculate acoustic wavelength."""
        return self.c / self.frequency

    @property
    def pitch(self):
        """
        Set element pitch.

        Made large for demonstration of spherical wavefronts,
        grating lobes not an issue in this demo
        """
        return 1.0

    @property
    def x_range(self):
        """Lateral extent of wavefronts, relative source."""
        return np.linspace(
            -self.x_lim,
            self.x_lim,
            self.n_points,
        )

    @property
    def delay_diff(self):
        """Find delay between elements."""
        return self.pitch * np.sin(self.steering_angle) / self.c

    @property
    def delay(self):
        """Find delay vector."""
        delay = self.delay_diff * np.arange(self.n_elements)
        delay -= min(delay)

        return delay

    @property
    def n_time_steps(self):
        return self.duration / self.time_step

    @property
    def radius_time_steps(self):
        steps = np.arange(self.n_time_steps) / self.n_time_steps
        return steps * self.z_max / np.cos(self.steering_angle)

    @property
    def x_sources(self):
        """Elements positions along x-axis, z=0."""
        rel_positions = np.arange(self.n_elements) - (self.n_elements - 1) / 2

        return self.pitch * rel_positions

    def calculate_wavefront(self, x0, radius):
        """
        Calculate wavefront as a circular arc around an element.
        x is lateral dimansion (y-axis).
        z is axial dimension (depth) (x-axis).

        Parameters
        ----------
        x0 : float
            Lateral position [m]
        radius : float
            Radius of wavefront

        Returns
        -------
        x : 1D NumPy array
            Lateral coordinate of wavefron curve
        """
        if radius >= 0:
            z_sq = radius**2 - self.x_range**2

            z = np.full_like(z_sq, np.nan, dtype=float)
            np.sqrt(z_sq, out=z, where=(z_sq >= 0))
        else:
            z = np.full_like(self.x_range, np.nan)

        return z, self.x_range + x0

    def draw_wavefronts(self, x_sources, radii):
        """Draw wavefronts around sources in lateral xy-plane."""

        for x0, radius, graph in zip(
            x_sources, radii, self.graphs["wavefronts"]
        ):
            z, x = self.calculate_wavefront(x0, radius)
            graph.set_data(z, x)

    def display_wavefront(self, r0=0):
        """Create figure and draw wavefronts."""
        radii = r0 - self.c * self.delay

        self.draw_wavefronts(self.x_sources, radii)

    def animate_wavefronts(self):
        """Run animation ow wavefronts."""

        self.update_wavefield_graphs()

        for radius in self.radius_time_steps:
            t0 = time.perf_counter()
            self.display_wavefront(radius)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            elapsed = time.perf_counter() - t0
            time.sleep(max(0, self.time_step - elapsed))

    def update_wavefield_graphs(self):
        """Remove old wavefront lines and create new set of empty lines."""
        ax = self.axes["wavefronts"]

        # Draw sources
        self.graphs["sources"].set_data(
            np.zeros_like(self.x_sources),
            self.x_sources,
        )

        # Indicator for direction
        self.graphs["direction"].set_data(
            [0, self.z_max],
            [0, self.z_max * np.sin(self.steering_angle)],
        )

        # Create new empty wavefront graphs
        for graph in self.graphs["wavefronts"]:
            graph.remove()

        self.graphs["wavefronts"] = [
            ax.plot([], [], **LINEFORMAT["wavefront"])[0]
            for _ in range(self.n_elements)
        ]
        ax.set_axis_on()

    def _initialise_graphs(self):
        """Initialise result graphs."""
        plt.close(FIGURE_NAME)

        fig, axes = plt.subplot_mosaic(
            [
                [".", "wavefronts", "wavefronts", "wavefronts"],
                ["logo", "wavefronts", "wavefronts", "wavefronts"],
            ],
            figsize=(16, 6),
            layout="constrained",
            num=FIGURE_NAME,
        )

        self._create_logo(axes["logo"])
        graphs = self._create_wavefront_plot(axes["wavefronts"])

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
            xlabel="Distance [m]",
            ylabel="Lateral position",
            title="Wavefronts ",
            aspect="equal",
            xlim=(0, self.z_max),
            ylim=self.x_ax_range,
        )

        ax.grid(visible=False)
        ax.set_facecolor("#F0FBFF")
        ax.set_axis_off()

        graphs = {}

        # Wavefront image
        graphs["wavefronts"] = [
            ax.plot([], [], **LINEFORMAT["wavefront"])[0]
            for _ in range(self.n_elements)
        ]

        # Acoustic axis, normally not changed
        graphs["axis"] = ax.plot(
            [0, self.z_max],
            [0, self.z_max * np.sin(self.steering_angle)],
            **LINEFORMAT["indicator"],
        )[0]

        # Sources, steering direction, changed during runtime
        graphs["sources"] = ax.plot([], [], **LINEFORMAT["source"])[0]
        graphs["direction"] = ax.plot([], [], **LINEFORMAT["indicator"])[0]

        return graphs

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
