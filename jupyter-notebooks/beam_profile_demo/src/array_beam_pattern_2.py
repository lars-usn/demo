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
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import MultipleLocator
import ipywidgets as widgets
from pathlib import Path

# Internal libraries
import beamplot_utilities as bpu

COLOR = {
    "text_face": "#F0FBFF",  # "#E6F3F7", " # "#F0FBFF", "#EAF7FA"
    "text_edge": "#7AA6B8",
    "intensity_background": "black",
}

LINEFORMAT = {
    "main": {"color": "C0", "linestyle": "solid"},
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

        # Display settings
        self.theta_max = 90  # deg  Max. angle to calculate
        self.n_x = 300  # No. of points in the x-direction (azimuth)
        self.n_z = 400  # No. of points in the z-direction (depth)
        self.db_range = 60  # dB   Dynamic gange on dB-scales
        self.db_gain = 6  # dB   Max. on dB-scales
        self.db_polar = 30  # dB   Dynamic range on polar plot

        # Grid and display settings, normally fixed
        self.x_max = 70.0  # m    Max. lateral dimension to calculate
        self.z_max = 140.0  # m    Max. depth to calculate
        self.theta_max = 90  # deg  Max. angle to calculate
        self.colormap = "inferno"

        self.z_axis = np.linspace(1, self.z_max, 400)
        self.x_axis = np.linspace(-self.x_max, self.x_max, 201)

        self.zx_plane = np.meshgrid(self.z_axis, self.x_axis)
        self.r, self.theta = self.calculate_axial_distance_angle()

        # Initialise figures and values
        self.fig, self.axes, self.graphs = self._initialise_graphs()
        self.update_values()
        self.update_intensity_scale()
        self.scale_axes()

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
    def wavelength(self):
        """Calculate acoustic wavelength."""
        return self.c / self.frequency

    @property
    def delay(self):
        """Delay between elements (no focusing)."""
        return -np.sin(self.steering_angle) * self.pitch / self.c

    @property
    def elements(self):
        """Make array of alemen numbers."""
        return np.arange(0, self.n_elements) + 1

    @property
    def delays(self):
        """Calculate delay for all elements."""
        nc = 1 / 2 * (self.n_elements + 1)
        tau = -(self.elements - nc) * self.delay
        return tau

    @property
    def width(self):
        """Width of element."""
        return self.pitch - self.kerf

    @property
    def aperture_width(self):
        """Width of aperture."""
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
        """Directivity of arrray of points."""
        s_theta = np.sin(theta) - np.sin(self.steering_angle)
        x = pi * self.pitch_lambda * s_theta
        x[x == 0] = 1e-4  # Avoid 0/0 erroers
        d = np.sin(self.n_elements * x) / (self.n_elements * np.sin(x))

        return d

    def p_axial(self):
        """Calculate axial pressure field in the azimuth plane (zx)."""
        r, theta = self.calculate_axial_distance_angle()

        p0 = self.n_elements * self.width / r
        p1 = self.directivity_element(theta)
        pn = self.directivity_array_points(theta)

        return p0 * p1 * pn

    # === Commands =============================
    def update_values(self):
        """Update graph values, no scale or other changes."""

        # Beam profile, intensity plot
        p_axial = self.p_axial()
        p_max = np.max(np.abs(p_axial))
        p_db = self.db(p_axial, reference=p_max)

        p_display = p_db.transpose()
        self.graphs["axial"].set_array(p_display.ravel())

        # Delays, stem-plot
        markerline, stemlines, baseline = self.graphs["delay"]
        n = self.elements
        tau = self.delays * 1e6
        markerline.set_data(n, tau)

        stemlines.set_segments([[[xi, 0], [xi, yi]] for xi, yi in zip(n, tau)])

        # self.graphs["delay"].set_data(self.elements, self.delays * 1e6)

        # Lateral beam profile
        # x = self.x_axis
        # z = self.z_axis
        # k_ref = np.argmin(abs(z - self.reference_distance))
        # p = p_axial[:, k_ref]
        # p_db = self.db(p, reference=p_max)
        # self.graphs["beamprofile"].set_data(x, p_db)

        # Update messages
        resulttext = self.update_resulttext()
        self.graphs["text"].txt.set_text(resulttext)

    def update_intensity_scale(self):
        """Update intensity graph levels."""
        self.graphs["axial"].set_clim(self.db_scale)
        self.graphs["colorbar"].set_ticks(self.db_ticks)

    def scale_axes(self):
        """
        Change scales of all graphs.

        Normally fixed at start and not changed when parameters are changed.
        """

    def update_resulttext(self):
        """
        Text box for lateral profile results.

        Returns
        -------
        str
            Formatted text with transducer beam parameters
        """
        header = (
            f"Frequency  $f$ = {self.frequency/1e3:.0f} kHz\n"
            rf"Wavelength  $\lambda$ = {self.wavelength*1e3:.1f} mm"
        )

        array_text = (
            r"No. of elements $N_{el}$ = "
            f"{self.n_elements:d} "
            "\n"
            f"Element width $w$ = {self.width*1e3:.2f}"
            "\n"
            f"Kerf $w$ = {self.kerf*1e3:.3f} mm"
            "\n"
            f"Pitch $d$ = {self.pitch*1e3:.2f} mm"
            f" = {self.pitch_lambda:.2f}"
            r" $\lambda$"
            "\n"
            f"Array width $D$ = {self.aperture_width*1e3:.0f} mm"
            f" = {self.aperture_lambda:.1f}"
            r" $\lambda$"
        )

        angle_text = (
            r"Steering angle $\theta_s$ = "
            rf"{self.steering_angle:.1f}$^\circ$"
        )

        distance_text = (
            r"Rayleigh distance $z_R$ = " f"{self.rayleigh_distance:.2f} m"
        )

        result_text = (
            header
            + "\n"
            + array_text
            + "\n"
            + angle_text
            + "\n"
            + distance_text
        )

        return result_text

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
            self.update_intensity_scale()

    # === Non-public methods ==========================================
    # Graphs and results
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
        Matplotlib AnchoredText
            Handle to text box
        """
        ax.axis("off")

        # Create empty anchored text box
        at = AnchoredText(
            "Beam parameters coming here",
            loc="upper center",
            pad=0.4,
            borderpad=0.2,
            frameon=True,
        )

        at.patch.set_facecolor(COLOR["text_face"])
        at.patch.set_edgecolor(COLOR["text_edge"])
        at.patch.set_boxstyle("round")

        ax.add_artist(at)

        return at

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
        print("Creating axial intensity plot")

        x_coords = self.x_axis
        y_coords = self.z_axis
        dummy_data = np.full((len(y_coords), len(x_coords)), np.nan)
        graph = ax.pcolormesh(
            x_coords,
            y_coords,
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
        matplotlib.lines.Line2D
            Line object that will be updated with delay data.
        """
        delay_max = 1e6 * self.aperture_width / (2 * self.c)

        bin_log = int(np.floor(np.log2(self.n_elements)))
        sep = max(1, 2 ** (bin_log - 2))
        delay_ticks = np.arange(0, self.n_elements + 1, sep)

        ax.set(
            xlim=(0, self.n_elements + 1),
            xticks=delay_ticks,
            ylim=(-delay_max, delay_max),
            xlabel="Element no.",
            ylabel=r"Delay [$\mu$s]",
            title="Element delays",
        )

        ax.grid(True, which="major", linewidth=0.8)
        ax.grid(True, which="minor", linewidth=0.3, alpha=0.3)

        #        (graph,) = ax.stem([], [], **STEM["main"])
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
            xlabel="Distance [m]",
            ylabel="Power [dB re. max]",
            title="Lateral beam profile",
        )

        ax.grid(
            visible=True,
            which="major",
            axis="x",
        )

        (graph,) = ax.plot([], [], **LINEFORMAT["main"])

        return graph

    def _initialise_graphs(self):
        """Initialise result graphs."""
        plt.close(FIGURE_NAME)

        fig, axes = plt.subplot_mosaic(
            [
                [".", "delay", "axial", "axial"],
                [".", "beamprofile", "axial", "axial"],
                ["text", "beamprofile", "axial", "axial"],
                ["logo", "beamprofile", "axial", "axial"],
            ],
            figsize=(14, 6),
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
        """
        Create widgets for interactive operation.

        Returns
        -------
        widget_layout : ipywidgets widget box
            Widget layout for use in Jupyter Notebook
        widget_list : dict of widgets
            Widgets for use in Jupyter Notebook
        """
        title = "Beam-profile from Single Element Transducer"
        title_widget = widgets.Label(
            title,
            style=dict(font_weight="bold"),
        )

        left_layout = {
            "continuous_update": True,
            "layout": widgets.Layout(width="95%"),
            "style": {"description_width": "50%"},
        }

        right_layout = {
            "continuous_update": True,
            "layout": widgets.Layout(width="95%"),
            "style": {"description_width": "30%"},
        }

        left_width = "25%"
        right_width = "75%"

        # Left column widgets (Dropboxes, number boxes)
        shape_widget = widgets.Dropdown(
            options=[
                ("Rectangular", False),
                ("Circular", True),
            ],
            value=True,
            description="Shape",
            **left_layout,
        )

        orientation_widget = widgets.Dropdown(
            options=[
                ("Azimuth (width)", True),
                ("Elevation (height)", False),
            ],
            value=True,
            description="Orientation",
            **left_layout,
        )

        db_range_widget = widgets.BoundedFloatText(
            value=self.db_range,
            min=6,
            max=120,
            step=6,
            description="Range [dB]",
            **left_layout,
        )

        db_gain_widget = widgets.BoundedFloatText(
            value=self.db_gain,
            min=-120,
            max=120,
            step=6,
            description="Gain [dB]",
            **left_layout,
        )

        left_col = widgets.VBox(
            [
                shape_widget,
                orientation_widget,
                db_range_widget,
                db_gain_widget,
            ],
            layout=widgets.Layout(width=left_width),
        )

        # Right column widgets (Sliders)
        frequency_widget = widgets.FloatSlider(
            value=self.frequency / 1e3,
            min=1,
            max=400,
            step=1,
            readout_format=".0f",
            description="Frequency [kHz]",
            **right_layout,
        )

        width_widget = widgets.FloatSlider(
            value=self.width * 1e3,
            min=10,
            max=400,
            step=10,
            readout_format=".0f",
            description="Width / Diameter [mm]",
            **right_layout,
        )

        height_widget = widgets.FloatSlider(
            value=self.height * 1e3,
            min=10,
            max=400,
            step=10,
            readout_format=".0f",
            description="Height [mm]",
            **right_layout,
        )

        distance_widget = widgets.FloatSlider(
            value=self.distance,
            min=1.0,
            max=self.z_max,
            step=1.0,
            readout_format=".0f",
            description="Distance [m]",
            **right_layout,
        )

        right_col = widgets.VBox(
            [
                frequency_widget,
                width_widget,
                height_widget,
                distance_widget,
            ],
            layout=widgets.Layout(width=right_width),
        )

        widget_layout = widgets.HBox(
            [left_col, right_col], layout=widgets.Layout(width="80%")
        )

        widget_layout = widgets.VBox([title_widget, widget_layout])

        widget_list = {
            "circular": shape_widget,
            "azimuth": orientation_widget,
            "db_range": db_range_widget,
            "db_gain": db_gain_widget,
            "frequency": frequency_widget,
            "width": width_widget,
            "height": height_widget,
            "distance": distance_widget,
        }

        return widget_layout, widget_list
