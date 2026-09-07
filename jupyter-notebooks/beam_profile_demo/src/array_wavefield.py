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

LINEFORMAT = {
    "wavefront": {"color": "C0", "linestyle": "dotted"},
    "reference": {"color": "C0", "linestyle": "dashed"},
    "source": {"color": "C1", "linestyle": "none", "marker": "o"}
}

LOGOFILE = "usn-logo-purple.png"
FIGURE_NAME = "Transducer Array Beamprofile"


class ArrayWaves:
    def __init__(self):
        self.pitch = 0.5
        self.n_elements = 6

        self.x_lim = 4.0
        self.z_max = 20.0
        self.x_ax_range = (-10.0, 10.0)
        self.n_points = 300

        self.x_range = np.linspace(
            -self.x_lim,
            self.x_lim,
            self.n_points,
        )

        # Initialise figures and values
        self.fig, self.axes, self.graphs = self._initialise_graphs()
        self.update_wavefield_graphs()

    @property
    def x_sources(self):
        """Elements positions anlon x-axis, z=0."""
        rel_positions = np.arange(self.n_elements) - self.n_elements/2

        return self.pitch * rel_positions

    def calculate_wavefront(self, x0, radius):
        """Calculate wavefront as circle sector around element.

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
        z_sq = radius**2 - self.x_range**2

        z = np.full_like(z_sq, np.nan, dtype=float)
        np.sqrt(z_sq, out=z, where=(z_sq >= 0))

        return z, self.x_range + x0

    def draw_wavefronts(self, x_sources, radii):
        """Draw wavefronts around sources in lateral xy-plane."""

        for x0, radius, graph in zip(
                x_sources,
                radii,
                self.graphs["wavefronts"],
        ):
            z, x = self.calculate_wavefront(x0, radius)
            graph.set_data(z, x)

    def make_wavefront_figure(self):
        """Create figure and draw wavefronts."""
        radii = 15 - 0.1*np.arange(self.n_elements)

        self.draw_wavefronts(self.x_sources, radii)

    def update_wavefield_graphs(self):
        """Remove old wavefront lines and create new set of empty lines."""
        for graph in self.graphs["wavefronts"]:
            graph.remove()

        self.graphs["wavefronts"] = [
            self.axes["wavefronts"].plot([], [], **LINEFORMAT["wavefront"])[0]
            for _ in range(self.n_elements)]

        z_pos = np.zeros_like(self.x_sources)
        self.axes["wavefronts"].plot(z_pos, self.x_sources,
                                     **LINEFORMAT["source"])

        self.axes["wavefronts"].axes("off")

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

        graphs = {}
        self._create_logo(axes["logo"])
        graphs["wavefronts"] = self._create_wavefront_plot(axes["wavefronts"])

        return fig, axes, graphs

    def _create_wavefront_plot(self, ax):
        """
        Create axis for beam profile graphs.

        Parameters
        ----------
        ax : Axis object
            Axis where wavefronts are plotted

        Returns
        -------
            Matplotlib Line2D
        """
        ax.set(
            xlabel="Distance [m]",
            ylabel="Lateral position",
            title="Wavefronts ",
            aspect="equal",
            xlim=(0, self.z_max),
            ylim=self.x_ax_range
        )

        ax.grid(visible=False)
        graph = [ax.plot([], [], **LINEFORMAT["wavefront"])[0]
                 for _ in range(self.n_elements)]

        return graph

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
