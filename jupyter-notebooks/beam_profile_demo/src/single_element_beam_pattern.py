"""Run this to import libraries and define internal functions."""

# Libraries
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.offsetbox import AnchoredText
import ipywidgets as widgets
from pathlib import Path

# Internal libraries
import beamplot_utilities as bpu

COLOR = {
    "transducer": "#A63D1F",  # "#B64926"  "#A63D1F" "#B35A1F" "#8C2D19"
    "transducer_background": "#F0FBFF",  # "#D6EFFC "#C2E7F7" "#E0F4FC"
    "text_face": "#F0FBFF",  # "#E6F3F7", " # "#F0FBFF", "#EAF7FA"
    "text_edge": "#7AA6B8",
    "contour": "white",
    "orientation_line": "#1F77B4",  # "#1F77B4" "#006BA4" "#2A6FBB" "#0072B2"
    "intensity_background": "black",
}

LINE = {
    "contour": {
        "colors": COLOR["contour"],
        "linestyles": "dotted",
        "alpha": 0.7,
    },
    "angle": {"color": COLOR["contour"], "linestyle": "dotted", "alpha": 0.7},
    "orientation": {
        "color": COLOR["orientation_line"],
        "linestyle": "dotted",
        "alpha": 1.0,
    },
    "indicator": {"color": "C1", "linestyle": "solid"},
    "main": {"color": "C0", "linestyle": "solid"},
}

LOGOFILE = "usn-logo-purple.png"
FIGURE_NAME = "Single Element Beamprofile"


class Transducer:
    """Define, calculate, and display a transducer beam profile."""

    def __init__(self, create_widgets=False):

        # Transducer definition
        self.circular = True  # Circular or rectangular element
        self.azimuth = True  # Show azimuth (x) or elevation (y) profile
        self.db_gain = 6  # dB   Max. on dB-scales
        self.db_range = 60  # dB   Dynamic range on dB-scales
        self.frequency = 100e3  # Hz  Ultrasound frequency
        self.width = 100e-3  # m   Element width (azimuth, x) or diameter
        self.height = 200e-3  # m   Element height (elevation, y)
        self.c = 1500  # m/s Speed of sound in load medium

        # Parameters used in calculations
        self.distance = 20.0  # m    Reference distance
        self.y_lim = 0.5  # Relative limit for beamwidth
        self.lim_text = "-6 dB"  # Text for markers

        # To be calculated during runtime
        self.beamwidth = np.nan
        self.x_sidelobe = np.nan
        self.y_sidelobe = np.nan
        self.db_sidelobe = np.nan

        # Grid and display settings, normally fixed
        self.z_max = 100.0  # m    Max. depth to calculate
        self.x_max = 12.0  # m    Max. lateral dimension to calculate
        self.d_max = 200e-3  # m    Max. dimension on element display
        self.colormap = "inferno"

        self.z_axis = np.linspace(1, self.z_max, 400)
        self.x_axis = np.linspace(-self.x_max, self.x_max, 201)

        self.zx_plane = np.meshgrid(self.z_axis, self.x_axis)
        self.r, self.theta = self.calculate_axial_distance_angle()

        self.xy_plane = np.meshgrid(self.x_axis, self.x_axis)
        self.rho = self.calculate_lateral_distance()

        # Initialise figures and values
        self.fig, self.axes, self.graphs = self._initialise_graphs()
        self.update_transducer_illustration()
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

    def calculate_lateral_distance(self):
        """
        Lateral distance from axis.

        Returns
        -------
        2D array of float
            Distance from axis to point (x,y) in the lateral plane
        """
        x, y = self.xy_plane
        return np.hypot(x, y)

    # Simple parameters are properties
    @property
    def wavelength(self):
        """Calculate acoustic wavelength."""
        return self.c / self.frequency

    @property
    def width_lambda(self):
        """Aperture width relative to wavelength."""
        return self.width / self.wavelength

    @property
    def height_lambda(self):
        """Aperture height relative to wavelength."""
        return self.height / self.wavelength

    @property
    def opening_angle(self):
        """Calculate opening angle from theory, two-sided, -6 dB."""
        if self.circular:
            x_6 = 0.705  # 6 dB limit, circular aperture
        else:
            x_6 = 0.603  # 6 dB limit, line (rectangular) aperture

        if self.azimuth or self.circular:
            d = self.width
        else:
            d = self.height

        arg = np.clip(x_6 * self.wavelength / d, -1, 1)
        return 2 * np.arcsin(arg)

    @property
    def rayleigh_distance(self):
        """Rayleigh distance, far-field limit."""
        if self.circular:
            d = self.width
        else:
            d = max(self.width, self.height)

        return d**2 / (2 * self.wavelength)

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
    def reference_distance(self):
        """Limit reference distance to outside far-field limit."""
        return max(self.distance, 1.1 * self.rayleigh_distance)

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
            Amplitude or power values
        reference : float
            Reference value for dB calculation
        power : bool
            Interpret x values as power (True) or amplitude (False)
        """
        x = np.asarray(x)

        if reference is None:
            reference = np.nanmax(x)

        reference = np.clip(abs(reference), 1e-20, None)

        scale = 20 if not power else 10
        arg = np.clip(np.abs(x), 1e-20, None)

        return scale * np.log10(arg / reference)

    def jinc(self, x):
        """jinc-function, Bessel-version of sinc, 2 J_1(pi x)/(pi x)."""
        x = np.asarray(x)

        with np.errstate(divide="ignore", invalid="ignore"):
            j = 2 * sp.j1(np.pi * x) / (np.pi * x)

        return np.where(np.abs(x) > 1e-10, j, 1.0)

    # === Axial plane ===================
    def p_azimuth(self):
        """
        Calculate pressure field in the azimuth plane.

        Returns
        -------
        2D array of float
            Pressure amplitude (signed) at point (z,x) in the azimuth plane
        """
        if self.circular:
            return self.p_circular()
        else:
            return self.p_line(self.width_lambda)

    def p_elevation(self):
        """
        Calculate pressure field in the elevation plane.

        Returns
        -------
        2D array of float
            Pressure amplitude (signed) at point (z,y) in the elevation plane
        """
        if self.circular:
            return self.p_circular()
        else:
            return self.p_line(self.height_lambda)

    def p_circular(self):
        """
        Calculate pressure field from a circular aperture.

        Returns
        -------
        2D array of float
            Pressure amplitude (signed) at point (z,x) in the axial plane
        """
        r = self.r
        theta = self.theta
        arg = self.width_lambda * np.sin(theta)
        p = 1 / r * self.jinc(arg)
        p[:, : self.rayleigh_index] = np.nan
        return p

    def p_line(self, aperture_lambda):
        """
        Calculate pressure field from a line aperture.

        Returns
        -------
        2D array of float
            Pressure amplitude (signed) at point (z,x)
        """
        r = self.r
        theta = self.theta
        arg = aperture_lambda * np.sin(theta)
        p = np.sinc(arg) / r
        p[:, : self.rayleigh_index] = np.nan
        return p

    # === Lateral plane ===================
    def calculate_lateral_distance_angles(self):
        """
        Calculate geometrical properties for the lateral plane.

        Lateral coordinates are fixed, total distance and angles depend
        on axial distance.

        Returns
        -------
        r : 2D array of float
            Distance from origo to point (x,y,z)
        theta : 2D array of float
            Azimuthal angle from origo to point (x,y,z)
        phi : 2D array of float
            Elevational angle from origo to point (x,y,z)
        gamma : 2D array of float
            Angle with acoustic axis from origo to point (x,y,z)
        """
        x, y = self.xy_plane
        rho = self.rho

        r = np.hypot(rho, self.reference_distance)
        theta = np.arctan2(x, self.reference_distance)  # Azimuth angle
        phi = np.arctan2(y, self.reference_distance)  # Elevation angle
        gamma = np.arctan2(rho, self.reference_distance)  # Angle with axis

        return r, theta, phi, gamma

    def p_lateral(self):
        """
        Calculate lateral amplitude at reference distance.

        Returns
        -------
        2D array of float
            Pressure amplitude (signed) at point (x,y,z)
        """
        r, theta, phi, gamma = self.calculate_lateral_distance_angles()

        if self.circular:
            arg = self.width_lambda * np.sin(gamma)
            p = 1 / r * self.jinc(arg)
        else:
            arg_az = self.width_lambda * np.sin(theta)
            arg_el = self.height_lambda * np.sin(phi)

            p_az = np.sinc(arg_az)
            p_el = np.sinc(arg_el)
            p = 1 / r * p_az * p_el

        return p

    # === Commands =============================
    def update_transducer_illustration(self):
        """Update aperture illustration, shape and dimensions."""
        w_mm = self.width * 1e3
        h_mm = self.height * 1e3

        patch_type = patches.Circle if self.circular else patches.Rectangle

        # Redraw shape if changed
        if not isinstance(self.graphs["transducer"], patch_type):
            self.graphs["transducer"].remove()

            if self.circular:
                self.graphs["transducer"] = patches.Circle(
                    (0, 0),
                    w_mm / 2,
                    fill=True,
                    color=COLOR["transducer"],
                )
            else:
                self.graphs["transducer"] = patches.Rectangle(
                    (-w_mm / 2, -h_mm / 2),
                    w_mm,
                    h_mm,
                    fill=True,
                    color=COLOR["transducer"],
                )

            self.axes["transducer"].add_patch(self.graphs["transducer"])

        # Update dimensions
        if self.circular:
            self.graphs["transducer"].set_radius(w_mm / 2)
        else:
            self.graphs["transducer"].set_xy((-w_mm / 2, -h_mm / 2))
            self.graphs["transducer"].set_width(w_mm)
            self.graphs["transducer"].set_height(h_mm)

        # Turn on correct orientation lines
        visible = self.azimuth

        for g in self.graphs["azimuth"]:
            g.set_visible(visible)
        for g in self.graphs["elevation"]:
            g.set_visible(not visible)

    def update_values(self):
        """Update graph values, no scale or other changes."""

        # Intensity plots
        p_az = self.p_azimuth()
        p_el = self.p_elevation()

        p_max = max(
            np.nanmax(np.abs(p_az)),
            np.nanmax(np.abs(p_el)),
        )

        # Axial
        p_axial = p_az if self.azimuth else p_el

        p_db = self.db(p_axial, reference=p_max)
        self.graphs["axial"].set_array(p_db.ravel())

        self.graphs["refline"].set_xdata(
            [self.reference_distance, self.reference_distance]
        )

        # Lateral
        p_db = self.db(self.p_lateral(), reference=p_max)
        self.graphs["lateral"].set_array(p_db.ravel())
        self.axes["lateral"].set_title(
            f"Lateral plane at {self.reference_distance:.0f} m"
        )

        # Lateral beam profile
        x = self.x_axis
        z = self.z_axis
        k_ref = np.argmin(abs(z - self.reference_distance))
        p = p_axial[:, k_ref]
        p_db = self.db(p, reference=p_max)
        self.graphs["beamprofile"].set_data(x, p_db)

        # Find reference values
        curve_analysis = bpu.AnalyseCurve(argument=x, value=p)
        xl, _ = curve_analysis.ref_values(
            y_rel=self.y_lim
        )  # Beam width limits
        self.beamwidth = xl[1] - xl[0]

        self.x_sidelobe, self.y_sidelobe = curve_analysis.sidelobe()

        self.db_sidelobe = float(
            self.db(
                self.y_sidelobe,
                reference=np.nanmax(p),
            ),
        )

        # Update messages
        self.update_resulttext()

    def update_intensity_scale(self):
        """Update intensity graph levels."""
        for g in (self.graphs["axial"], self.graphs["lateral"]):
            g.set_clim(self.db_scale)

        self.graphs["colorbar"].set_ticks(self.db_ticks)

        self.axes["beamprofile"].set(ylim=self.db_scale, yticks=self.db_ticks)

        self.axes["beamprofile"].grid(
            visible=True,
            which="major",
            axis="y",
        )

    def scale_axes(self):
        """
        Change scales of all graphs.

        Normally fixed at start and not changed when parameters are changed.
        """
        ax = self.axes

        axis_max = self.d_max * 1e3 * np.array([-1, 1])
        ax["transducer"].set(
            xlim=axis_max,
            ylim=axis_max,
        )

        lateral_max = self.x_max * np.array([-1, 1])
        ax["axial"].set(
            ylim=lateral_max,
            xlim=[0, self.z_max],
        )

        ax["lateral"].set(
            xlim=lateral_max,
            ylim=lateral_max,
        )

        ax["beamprofile"].set(xlim=lateral_max)

    # === Non-public methods ==========================================
    def _create_transducer_illustration(self, ax):
        """
        Create colored patch to illustrate transducer.

        Parameters
        ----------
        ax : Axis object
            Axis where transducer image is shown

        Returns
        -------
        Matplotlib patch
            Illustration of transducer element
        """

        ax.set(
            title="Transducer shape",
            facecolor=COLOR["transducer_background"],
            box_aspect=1,
            xlabel="Azimuth [mm]",
            ylabel="Elevation [mm]",
        )
        transducer_fill = {"fill": True, "color": COLOR["transducer"]}
        if self.circular:
            patch = patches.Circle(
                (0, 0),
                radius=0,
                **transducer_fill,
            )
        else:
            patch = patches.Rectangle(
                (0, 0),
                width=0,
                height=0,
                **transducer_fill,
            )

        ax.add_patch(patch)
        return patch

    def update_resulttext(self):
        """Update text box with array parameters."""

        LEGEND_COL = 0
        SYMBOL_COL = 1
        VALUE_COL = 2
        WAVELENGTH_COL = 3

        value_lines = [
            f"{self.frequency/1e3:.0f} kHz",
            f"{self.wavelength*1e3:.1f} mm",
            f"{self.width*1e3:.0f} mm",
            f"{self.height*1e3:.0f} mm",
            f"{self.rayleigh_distance:.1f} m",
            rf"{np.degrees(self.opening_angle):.1f}$^\circ$",
            f"{self.beamwidth:.2f} m",
            f"{self.db_sidelobe:.1f} dB",
        ]

        wavelength_lines = [
            "",
            "",
            rf"{self.width_lambda:.1f}$\lambda$",
            rf"{self.height_lambda:.1f}$\lambda$",
            "",
            "",
            "",
            "",
        ]

        legend_lines = [
            self.graphs["text"][(row, LEGEND_COL)].get_text().get_text()
            for row in range(len(value_lines))
        ]

        symbol_lines = [
            self.graphs["text"][(row, SYMBOL_COL)].get_text().get_text()
            for row in range(len(value_lines))
        ]

        if not self.circular:
            symbol_lines[2] = "w"
            symbol_lines[3] = "h"
            legend_lines[2] = "Width"
            legend_lines[3] = "Height"
        else:
            symbol_lines[2] = "D"
            symbol_lines[3] = ""
            legend_lines[2] = "Diameter"
            legend_lines[3] = ""
            value_lines[3] = ""
            wavelength_lines[3] = ""

        assert len(legend_lines) == len(value_lines) == len(wavelength_lines)

        for line_no, (
            legend_txt,
            symbol_txt,
            value_txt,
            wavelength_txt,
        ) in enumerate(
            zip(legend_lines, symbol_lines, value_lines, wavelength_lines)
        ):

            self.graphs["text"][(line_no, LEGEND_COL)].get_text().set_text(
                legend_txt
            )
            self.graphs["text"][(line_no, SYMBOL_COL)].get_text().set_text(
                symbol_txt
            )
            self.graphs["text"][(line_no, VALUE_COL)].get_text().set_text(
                value_txt
            )
            self.graphs["text"][(line_no, WAVELENGTH_COL)].get_text().set_text(
                wavelength_txt
            )

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
            ["Width", "$w$", "-", "-"],
            ["Height", "$h$", "-", "-"],
            ["Rayleigh distance ", r"$z_R$", "-", ""],
            ["Opening angle (-6 dB)", r"$\theta_0$", "-", ""],
            ["Beam width (-6 dB)", "$D_z$", "-", ""],
            ["Highest sidelobe", "$PSL$", "-", ""],
        ]

        table = ax.table(
            cellText=resulttext,
            loc="upper left",
            cellLoc="left",
            colWidths=[0.40, 0.20, 0.25, 0.25],
        )

        for cell in table.get_celld().values():
            cell.set_linewidth(0.2)
            cell.visible_edges = "TB"
            cell.set_facecolor(COLOR["text_face"])
            cell.PAD = 0.05
            cell.set_text_props(fontfamily="DejaVu Sans")

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.1)

        for r in range(len(resulttext)):
            table[(r, 1)].set_text_props(ha="center")
            table[(r, 2)].set_text_props(ha="left")

        return table

    # def _create_resulttextbox(self, ax):
    #     """
    #     Create and attach a formatted results text box to an Axes.

    #     The text box is anchored to an axis and remains fixed relative to
    #     the axes if the figure is resized.

    #     Parameters
    #     ----------
    #     ax : Axis object
    #         Axis where text is shown

    #     Returns
    #     -------
    #     Matplotlib AnchoredText
    #         Handle to text box
    #     """
    #     ax.axis("off")

    #     # Create empty anchored text box
    #     at = AnchoredText(
    #         "Beam parameters coming here",
    #         loc="upper center",
    #         pad=0.4,
    #         borderpad=0.2,
    #         frameon=True,
    #     )

    #     at.patch.set_facecolor(COLOR["text_face"])
    #     at.patch.set_edgecolor(COLOR["text_edge"])
    #     at.patch.set_boxstyle("round")

    #     ax.add_artist(at)

    #     return at

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
            aspect="equal",
            xlabel="Depth (z) [m]",
            ylabel="Azimuth / Elevation [m]",
            title="Axial plane",
            facecolor=COLOR["intensity_background"],
        )

        x_coords = self.z_axis
        y_coords = self.x_axis
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

    def _create_lateral_plot(self, ax):
        """Create axis for lateral intensity plots.

        Parameters
        ----------
        ax : Axis object
            Axis where lateral intensity image is shown

        Returns
        -------
        Matplotlib QuadMesh
            Handle to fill with intensity data
        """
        ax.set(box_aspect=1, xlabel="Azimuth [m]", ylabel="Elevation [m]")

        x, y = self.xy_plane
        x_coords = x[0, :]
        y_coords = y[:, 0]
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

    def _create_orientation_lines(self, axs):
        """
        Create azimuth and elevation reference lines.

        Parameters
        ----------
        axs : List of axis objects
            Axes where orientation lines are drawn

        Returns
        -------
        azimuth_lines : list of Matplotlib Line2D
            Handle to azimuth line data
        elevation_lines : list of Matplotlib Line2D
            Handle to elevation line data
        """

        azimuth_lines = []
        elevation_lines = []
        for ax in axs:
            azimuth_lines.append(
                ax.axhline(
                    y=0,
                    **LINE["orientation"],
                )
            )
            elevation_lines.append(
                ax.axvline(
                    x=0,
                    **LINE["orientation"],
                )
            )
        return azimuth_lines, elevation_lines

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

        (graph,) = ax.plot([], [], **LINE["main"])

        return graph

    def _initialise_graphs(self):
        """Initialise result graphs."""
        plt.close(FIGURE_NAME)

        fig, axes = plt.subplot_mosaic(
            [
                ["transducer", "axial", "axial"],
                ["transducer", "axial", "axial"],
                ["transducer", "axial", "axial"],
                ["text", "lateral", "beamprofile"],
                ["text", "lateral", "beamprofile"],
                ["logo", "lateral", "beamprofile"],
            ],
            figsize=(14, 6),
            layout="constrained",
            num=FIGURE_NAME,
        )

        graphs = {}
        self._create_logo(axes["logo"])
        graphs["transducer"] = self._create_transducer_illustration(
            axes["transducer"]
        )
        graphs["text"] = self._create_resulttextbox(axes["text"])
        graphs["axial"] = self._create_axial_plot(axes["axial"])
        graphs["lateral"] = self._create_lateral_plot(axes["lateral"])
        graphs["beamprofile"] = self._create_beamprofile_plot(
            axes["beamprofile"]
        )

        graphs["azimuth"], graphs["elevation"] = (
            self._create_orientation_lines(
                (axes["transducer"], axes["lateral"])
            )
        )

        # Reference line for distance
        graphs["refline"] = axes["axial"].axvline(
            x=self.reference_distance,
            **LINE["orientation"],
        )

        # Colorbar for intensity plots
        graphs["colorbar"] = fig.colorbar(
            graphs["axial"], ax=axes["axial"], label="dB re. max"
        )

        return fig, axes, graphs

    # === Widget callbacks =========================================
    def _frequency_change_callback(self, change):
        self.frequency = float(change["new"]) * 1e3
        self.update_values()

    def _shape_change_callback(self, change):
        self.circular = change["new"]
        self.update_transducer_illustration()
        self.update_values()

    def _orientation_change_callback(self, change):
        self.azimuth = change["new"]
        self.update_transducer_illustration()
        self.update_values()

    def _width_change_callback(self, change):
        self.width = float(change["new"]) * 1e-3
        self.update_transducer_illustration()
        self.update_values()

    def _height_change_callback(self, change):
        self.height = float(change["new"]) * 1e-3
        self.update_transducer_illustration()
        self.update_values()

    def _distance_change_callback(self, change):
        self.distance = float(change["new"])
        self.update_values()

    def _db_gain_change_callback(self, change):
        self.db_gain = change["new"]
        self.update_intensity_scale()

    def _db_range_change_callback(self, change):
        self.db_range = change["new"]
        self.update_intensity_scale()

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
        shape_widget.observe(
            self._shape_change_callback,
            names="value",
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
        orientation_widget.observe(
            self._orientation_change_callback,
            names="value",
        )

        db_range_widget = widgets.BoundedFloatText(
            value=self.db_range,
            min=6,
            max=120,
            step=6,
            description="Range [dB]",
            **left_layout,
        )
        db_range_widget.observe(
            self._db_range_change_callback,
            names="value",
        )

        db_gain_widget = widgets.BoundedFloatText(
            value=self.db_gain,
            min=-120,
            max=120,
            step=6,
            description="Gain [dB]",
            **left_layout,
        )
        db_gain_widget.observe(
            self._db_gain_change_callback,
            names="value",
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
        frequency_widget.observe(
            self._frequency_change_callback,
            names="value",
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
        width_widget.observe(
            self._width_change_callback,
            names="value",
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
        height_widget.observe(
            self._height_change_callback,
            names="value",
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
        distance_widget.observe(
            self._distance_change_callback,
            names="value",
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
