"""Calculate the beam pattern from a line array.

The array consists of N elements described either as points or rectangular
elements.
Results are calculated in the far-field of the array, using the
Fraunhofer-Approximation.

An interactive version can be run from the Jupyter Notebook 'array_demo.ipynb'
"""

# Libraries
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ipywidgets as widgets
from pathlib import Path

COLOR = {
    "text_face": "#F5F5F5",  # "#E6F3F7", " # "#F0FBFF", "#EAF7FA"
    "text_edge": "#7AA6B8",
    "intensity_background": "black",
}

LINEFORMAT = {
    "main": {"color": "C0", "linestyle": "solid"},
    "reference": {"color": "C0", "linestyle": "dashed"},
}

STEMFORMAT = {
    "main": {"linefmt": "C0", "basefmt": "#101010"},
}


LOGOFILE = "usn-logo-purple.png"
FIGURE_NAME = "Transducer Array Beamprofile"


class Array:
    """Define, calculate, and display a transducer beam profile."""

    def __init__(self, create_widgets=False):

        # Transducer definition
        self.n_elements = 32  # Number of elements in array
        self.kerf = 100e-6  # m   Kerf between elements
        self.pitch = 7.5e-3  # m   Pitch between elements
        self.frequency = 100e3  # Hz  Ultrasound frequency
        self.steering_angle = np.radians(20)  # rad Steering angle
        self.c = 1500  # m/s Speed of soundin load medium

        # Grid and display settings, normally fixed
        self.db_range = 60  # dB   Dynamic gange on dB-scales
        self.db_gain = 6  # dB   Max. on dB-scales
        self.db_polar = 30  # dB   Dynamic range on polar plot

        self.x_max = 70.0  # m    Max. lateral dimension to calculate
        self.z_max = 140.0  # m    Max. depth to calculate
        self.theta_max_degrees = 90  # deg  Max. angle to calculate
        self.colormap = "inferno"

        self.n_theta = 600  # No. of points in beam profile plot
        self.n_x = 601  # No. of points in the x-direction (azimuth)
        self.n_z = 600  # No. of points in the z-direction (depth)
        self.z_axis = np.linspace(
            1,
            self.z_max,
            self.n_z,
        )
        self.x_axis = np.linspace(
            -self.x_max,
            self.x_max,
            self.n_x,
        )
        self.theta_axis = np.linspace(
            -self.theta_max,
            self.theta_max,
            self.n_theta,
        )
        self.zx_plane = np.meshgrid(self.z_axis, self.x_axis)
        self.r, self.theta = self.calculate_axial_distance_angle()

        # Initialise figures and values
        self.fig, self.axes, self.graphs = self._initialise_graphs()
        self.update_values()
        self.scale_delay_plot()
        self.scale_intensity_plot()

        if create_widgets:
            self.widget_layout, self.widgets = self._create_widgets()

    # === Calculated parameters ===========================
    def calculate_axial_distance_angle(self):
        """
        Calculate distance and angles for axial plot.

        Returns
        -------
        r : 2D array of float
            Distance from origo to point (z,x) in the axial plane
        theta : 2D array of float
            Angle from origo to point (z,x) in the axial plane
        """
        z, x = self.zx_plane
        r = np.hypot(z, x)
        theta = np.arctan2(x, z)

        return r, theta

    # Simple parameters are properties
    @property
    def theta_max(self):
        """Max. angle to calculate."""
        return np.radians(self.theta_max_degrees)

    @property
    def wavelength(self):
        """Calculate acoustic wavelength."""
        return self.c / self.frequency

    @property
    def delay(self):
        """Delay between elements (no focusing)."""
        return -np.sin(self.steering_angle) * self.pitch / self.c

    @property
    def elements(self):
        """Elementr indices numbered from 1 to N."""
        return np.arange(0, self.n_elements) + 1

    @property
    def delays(self):
        """Calculate delay for all elements."""
        # nc = 1 / 2 * (self.n_elements + 1)  # Rel. array centre
        tau = -self.elements * self.delay

        return tau - np.min(tau)

    @property
    def width(self):
        """Width of element from pich and kerf."""
        return self.pitch - self.kerf

    @property
    def aperture_width(self):
        """Width of entire aperture."""
        return self.n_elements * self.pitch - self.kerf

    @property
    def pitch_lambda(self):
        """Pitch relative to wavelength."""
        return self.pitch / self.wavelength

    @property
    def width_lambda(self):
        """Element width relative to wavelength."""
        return self.width / self.wavelength

    @property
    def aperture_lambda(self):
        """Aperture width relative to wavelength."""
        return self.aperture_width / self.wavelength

    @property
    def rayleigh_distance(self):
        """Rayleigh distance, far-field limit."""
        return self.aperture_width**2 / (2 * self.wavelength)

    @property
    def rayleigh_index(self):
        """Find Rayleigh distance index."""
        index = np.searchsorted(
            self.z_axis,
            self.rayleigh_distance,
            side="right",
        )
        return index

    @property
    def db_scale(self):
        """Calculate dB-scale limits from gain and dynamic range."""
        return np.array([-self.db_range, 0]) - self.db_gain

    @property
    def db_ticks(self):
        """Set major dB ticks at fixed intervals (6 dB)."""
        db_sep = 6

        vmin, vmax = self.db_scale
        ticks = np.arange(
            db_sep * np.floor(vmin / db_sep),
            db_sep * np.ceil(vmax / db_sep) + db_sep,
            db_sep,
        )
        return ticks

    def db(self, x, reference=None, power=False):
        """
        Decibel from amplitude or power.

        Parameters
        ----------
        x : array of float
            Amplitue or power values
        reference : float
            Reference value for dB calculation
        power : bool
            Interpret x values as power (True) or amplitude (False)
        """
        x = np.asarray(x)

        if reference is None:
            reference = np.max(x)

        scale = 20 if not power else 10
        arg = np.clip(np.abs(x), 1e-20, None)

        return scale * np.log10(arg / reference)

    # === Axial plane ===================
    def directivity_element(self, theta):
        """Directivity of one element."""
        return np.sinc(self.width_lambda * np.sin(theta))

    def directivity_array_points(self, theta):
        """Directivity of array of points."""
        theta_diff = np.sin(theta) - np.sin(self.steering_angle)
        arg = pi * self.pitch_lambda * theta_diff
        arg = np.where(np.abs(arg) < 1e-12, np.copysign(1e-12, arg), arg)
        d = np.sin(self.n_elements * arg) / (self.n_elements * np.sin(arg))

        return d

    def p_axial(self):
        """Calculate axial pressure field in the azimuth plane (zx)."""
        r = self.r
        theta = self.theta

        p_aperture = self.n_elements * self.width / r
        p_aperture[:, : self.rayleigh_index] = 0

        p_element = self.directivity_element(theta)
        p_points = self.directivity_array_points(theta)

        return p_aperture * p_element * p_points

    # === Commands =============================
    def _update_stem_plot(self, graph, x, y, base=0):
        """Update values on a stem-plot.

        Parameters
        ----------
        graph : StemContainer
            Plot values returned from stem-command
        x : 1D array of float
            x-values
        y : 1D array of float
            y-values
        """
        markerline, stemlines, baseline = graph

        baseline.set_data(x, base * np.ones_like(y))
        markerline.set_data(x, y)
        stemlines.set_segments(
            [[[xi, base], [xi, yi]] for xi, yi in zip(x, y)]
        )

    def update_values(self):
        """Update graph values, no scale or other changes."""

        # Beam profile, intensity plot
        p_ref = self.n_elements * self.width
        p_db = self.db(self.p_axial(), reference=p_ref)
        p_display = p_db.transpose()
        self.graphs["axial"].set_array(p_display.ravel())

        # Delays, stem-plot
        self._update_stem_plot(
            self.graphs["delay"],
            self.elements,
            self.delays * 1e6,
        )

        # Beamprofile plot, polar
        theta = self.theta_axis
        p_element = self.directivity_element(theta)
        p_points = self.directivity_array_points(theta)
        p_array = p_element * p_points

        array_graph, element_graph = self.graphs["beamprofile"]
        array_graph.set_data(
            theta,
            self.db(p_array, reference=1),
        )
        element_graph.set_data(
            theta,
            self.db(p_element, reference=1),
        )

        # Information text
        self.update_resulttext()

    def scale_intensity_plot(self):
        """Update intensity graph levels."""
        self.graphs["axial"].set_clim(self.db_scale)
        self.graphs["colorbar"].set_ticks(self.db_ticks)

    def scale_delay_plot(self):
        """Scale axes on element delay plots."""
        delay_max = self.aperture_width / self.c
        delay_scale_max_um = np.ceil(1e6 * delay_max / 10) * 10

        bin_log = int(np.floor(np.log2(self.n_elements)))
        sep = max(1, 2 ** (bin_log - 2))
        delay_ticks = np.arange(0, self.n_elements + 1, sep)

        self.axes["delay"].set(
            xlim=(0, self.n_elements + 1),
            xticks=delay_ticks,
            ylim=(0, delay_scale_max_um),
        )

    def update_resulttext(self):
        """Update text box with array parameters."""

        value_lines = [
            f"{self.frequency/1e3:.0f} kHz",
            f"{self.wavelength*1e3:.1f} mm",
            f"{self.n_elements}",
            f"{self.width*1e3:.1f} mm",
            f"{self.kerf*1e3:.1f} mm",
            f"{self.pitch*1e3:.1f} mm",
            f"{self.aperture_width*1e3:.0f} mm",
            rf"{np.degrees(self.steering_angle):.0f}$^\circ$",
            f"{self.rayleigh_distance:.1f} m",
        ]

        lamda_symb = r"$\lambda$"
        wavelength_lines = [
            "",
            "",
            "",
            f"{self.width_lambda:.2f}" + lamda_symb,
            "",
            f"{self.pitch_lambda:.2f}" + lamda_symb,
            f"{self.aperture_lambda:.1f}" + lamda_symb,
            "",
            "",
        ]

        for line_no, value in enumerate(value_lines):
            self.graphs["text"][(line_no, 2)].get_text().set_text(value)

        for line_no, value in enumerate(wavelength_lines):
            self.graphs["text"][(line_no, 3)].get_text().set_text(value)

        return

    def interact(
        self,
        n_elements=None,
        freq_khz=None,
        pitch_mm=None,
        db_range=None,
        db_gain=None,
        steering_angle_degrees=None,
    ):
        """
        Scale inputs and  display the resulting response.

        For interactive operation with dimensions in mm and frequency in kHz.
          Existing values are retained if a parameter is omitted.

        Parameters
        ----------
        freq_khz: float, optional
            Frequency [kHz]
        pitch_mm: float, optional
            Array pitch [mm]. Distance between elements
        db_range: float
            Range on dB-axes
        db_gain: float
            Maximum on dB-axes
        steering_angle_degrees : float
            Steering angle [deg]
        """
        if n_elements is not None:
            self.n_elements = int(n_elements)

        if freq_khz is not None:
            self.frequency = float(freq_khz) * 1e3

        if pitch_mm is not None:
            self.pitch = float(pitch_mm) * 1e-3

        if steering_angle_degrees is not None:
            self.steering_angle = np.radians(steering_angle_degrees)

        if db_range is not None:
            self.db_range = db_range

        if db_gain is not None:
            self.db_gain = db_gain

        if any(
            v is not None
            for v in (n_elements, freq_khz, pitch_mm, steering_angle_degrees)
        ):
            self.update_values()

        if any(v is not None for v in (db_range, db_gain)):
            self.scale_intensity_plot()

        if any(v is not None for v in (n_elements, pitch_mm)):
            self.scale_delay_plot()

    # === Non-public methods ==========================================
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
            ["Frequency", "$f$", "-", ""],
            ["Wavelenght", r"$\lambda$", "-", ""],
            ["No. of elements", "$N_{el}$", "-", ""],
            ["Element width", "$w$", "-", "-"],
            ["Kerf", "$k$", "-", ""],
            ["Pitch", "$d$", "-", ""],
            ["Array width", "$D$", "-", ""],
            ["Steering angle", r"$\theta_s$", "-", ""],
            ["Rayleigh distance ", r"$z_R$", "-", ""],
        ]

        table = ax.table(
            cellText=resulttext,
            loc="upper left",
            cellLoc="left",
            colWidths=[0.45, 0.10, 0.25, 0.15],
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

    def _create_axial_plot(self, ax):
        """Create axis for axial intensity plot.

        Parameters
        ----------
        ax : Axis object
            Axis where axial intensity image shown

        Returns
        -------
        Matplotlib QuadMesh
            Handle to fill with intensity data
        """
        ax.set(
            xlim=(-self.x_max, self.x_max),
            ylim=(self.z_max, 0),
            aspect="equal",
            xlabel="Azimuth [m]",
            ylabel="Depth  [m]",
            title="Axial plane",
            facecolor=COLOR["intensity_background"],
        )

        dummy_data = np.full(
            (self.n_z, self.n_x),
            np.nan,
        )

        graph = ax.pcolormesh(
            self.x_axis,
            self.z_axis,
            dummy_data,
            clim=self.db_scale,
            cmap=self.colormap,
            shading="auto",
        )

        return graph

    def _create_delay_plot(self, ax):
        """
        Create axis for displaying element delays.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis where element delays are plotted.

        Returns
        -------
        StemContainer
            Stem plot object that will be updated with delay data.
        """
        ax.set(
            xlabel="Element no.",
            ylabel=r"Delay [$\mu$s]",
            title="Element delays",
        )

        ax.grid(True, which="major")
        ax.grid(True, which="minor")

        graph = ax.stem([0, 0], [1, 1], **STEMFORMAT["main"])

        return graph

    def _create_beamprofile_plot(self, ax):
        """
        Create axis for beam profile graphs.

        Parameters
        ----------
        ax : Axis object
            Axis where lateral intensity image is shown

        Returns
        -------
            Matplotlib Line2D
        """

        ax.set(
            thetamin=-90,
            thetamax=+90,
            rmin=-self.db_polar,
            rmax=0.0,
            rticks=[-20, -6, 0],
        )
        ax.set_theta_zero_location("S")  # Unreliable in ax.set()

        # Title as manually controlled text object,
        # works better with polar plot
        ax.text(
            0.5,
            0.90,
            "Radiation Diagram [dB re. max]",
            transform=ax.transAxes,
            ha="center",
            va="top",
        )

        (array_graph,) = ax.plot([], [], **LINEFORMAT["main"])
        (element_graph,) = ax.plot([], [], **LINEFORMAT["reference"])

        return (array_graph, element_graph)

    def _initialise_graphs(self):
        """Initialise result graphs."""
        plt.close(FIGURE_NAME)

        fig, axes = plt.subplot_mosaic(
            [
                [".", "delay", "axial", "axial"],
                [".", "delay", "axial", "axial"],
                ["text", "beamprofile", "axial", "axial"],
                ["text", "beamprofile", "axial", "axial"],
                ["logo", "beamprofile", "axial", "axial"],
            ],
            per_subplot_kw={"beamprofile": {"projection": "polar"}},
            figsize=(16, 6),
            layout="constrained",
            num=FIGURE_NAME,
        )

        graphs = {}
        self._create_logo(axes["logo"])
        graphs["text"] = self._create_resulttextbox(axes["text"])
        graphs["axial"] = self._create_axial_plot(axes["axial"])
        graphs["delay"] = self._create_delay_plot(axes["delay"])
        graphs["beamprofile"] = self._create_beamprofile_plot(
            axes["beamprofile"]
        )

        # Colorbar for intensity plots
        graphs["colorbar"] = fig.colorbar(
            graphs["axial"], ax=axes["axial"], label="dB re. max"
        )

        return fig, axes, graphs

    # Interactive widgets
    def _create_widgets(self):
        """Create widgets for interactive operation."""
        title = "Beam-profile from Transducer Array"
        title_widget = widgets.Label(title, style=dict(font_weight="bold"))

        text_layout = {
            "continuous_update": False,
            "layout": widgets.Layout(width="95%"),
            "style": {"description_width": "50%"},
        }

        slider_layout = {
            "continuous_update": True,
            "layout": widgets.Layout(width="95%"),
            "style": {"description_width": "30%"},
        }

        text_width = "20%"
        slider_width = "60%"

        # Text widgets (Dropboxes, number boxes)
        n_elements_widget = widgets.BoundedIntText(
            value=self.n_elements,
            min=1,
            max=256,
            step=1,
            description="No. of elements",
            **text_layout,
        )

        frequency_widget = widgets.BoundedFloatText(
            value=self.frequency / 1e3,
            min=10,
            max=400,
            step=1,
            description="Frequency [kHz]",
            **text_layout,
        )

        pitch_widget = widgets.BoundedFloatText(
            value=self.pitch * 1e3,
            min=0.1,
            max=100,
            step=0.1,
            description="Element pitch [mm]",
            **text_layout,
        )

        db_range_widget = widgets.BoundedFloatText(
            value=self.db_range,
            min=6,
            max=120,
            step=6,
            description="Range [dB]",
            **text_layout,
        )

        db_gain_widget = widgets.BoundedFloatText(
            value=self.db_gain,
            min=-120,
            max=120,
            step=6,
            description="Gain [dB]",
            **text_layout,
        )

        steering_angle_widget = widgets.FloatSlider(
            min=-90,
            max=90,
            value=0,
            step=1,
            readout_format=".0f",
            description="Steering angle [Deg.]",
            **slider_layout,
        )

        array_parameter_column = widgets.VBox(
            [
                frequency_widget,
                n_elements_widget,
                pitch_widget,
            ],
            layout=widgets.Layout(width=text_width),
        )

        scaling_column = widgets.VBox(
            [
                db_range_widget,
                db_gain_widget,
            ],
            layout=widgets.Layout(width=text_width),
        )

        slider_column = widgets.VBox(
            [steering_angle_widget],
            layout=widgets.Layout(width=slider_width),
        )

        widget_layout = widgets.HBox(
            [
                array_parameter_column,
                scaling_column,
                slider_column,
            ],
            layout=widgets.Layout(width="80%"),
        )

        widget_layout = widgets.VBox([title_widget, widget_layout])

        widget = {
            "n_elements": n_elements_widget,
            "frequency": frequency_widget,
            "pitch": pitch_widget,
            "db_range": db_range_widget,
            "db_gain": db_gain_widget,
            "steering_angle": steering_angle_widget,
        }

        return widget_layout, widget
