"""Run this to import libraries and define internal functions."""

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.offsetbox import AnchoredText
import ipywidgets as widgets
from pathlib import Path

# Internal libraries
import beamplot_utilities as bpu

COLOR = {
    "element": "#A63D1F",  # "#B64926"  "#A63D1F" "#B35A1F" "#8C2D19"
    "element_background": "#FDF0E3",  # "#FFF4E8 "#FDF0E3" "#FAE7D6" "#F5EBDD"
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


class WidgetLayout:
    """Container for widgets and layout for interactive Jupyter Notebook."""

    def __init__(self, layout, widget):
        self.layout = layout
        self.widget = widget


class Transducer:
    """Main class. Define, calculate, and display transducer beam profile."""

    def __init__(self, create_widgets=False):

        # Transducer definition
        self.circular = True  # Circular or rectangular element
        self.width = 100e-3  # m   Element width (azimuth, x) or diameter
        self.height = 200e-3  # m   Element height (elevation, y)
        self.frequency = 100e3  # Hz  Ultrasound frequency
        self.c = 1500  # m/s Speed of sound in load medium

        # Calculation settings
        self.distance = 20.0  # m    Reference distance
        self.y_lim = 0.5  # Relative limit for beamwidth
        self.lim_text = "-6 dB"  # Text for markers
        self.x_sidelobe = np.nan
        self.azimuth = True  # Show azimuth (x) or elevation (y) profile

        # Display settings
        self.theta_max = 90  # deg  Max. angle to calculate
        self.d_max = 200e-3  # m    Max. dimension on element display
        self.x_max = 15.0  # m    Max. lateral dimension to calculate
        self.z_max = 100.0  # m    Max. depth to calculate
        self.db_range = 60  # dB   Dynamic range on dB-scales
        self.db_gain = 6  # dB   Max. on dB-scales
        self.colormap = "inferno"

        # Initialisation
        self.fig, self.axes, self.graphs = self._initialise_graphs()
        self.update_element()
        self.update_values()
        self.update_intensity_scale()
        self.scale_axes()

        if create_widgets:
            self.widget = self._create_widgets()

    # === Calculated parameters ===========================
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

        return 2 * np.arcsin(x_6 * self.wavelength / d)

    @property
    def rayleigh_distance(self):
        """Rayleigh distance, far-field limit."""
        if self.circular:
            d = self.width
        else:
            d = max(self.width, self.height)

        return d**2 / (2 * self.wavelength)

    @property
    def reference_distance(self):
        """Limit reference distance to outside far-field limit."""
        return max(self.distance, 1.1 * self.rayleigh_distance)

    # === Axial plane ===================
    def x_axis(self):
        """Lateral dimension for axial plot (x or y)."""
        return np.linspace(-self.x_max, self.x_max, 201)


#### WRONG when updating only values. Must be fixed ============================
    def z_axis(self):
        """Axial dimension (depth) for axial plot (z)."""
        return np.linspace(1, self.z_max, 400)

    def zx_plane(self):
        """Axial plane (zx or zy) to plot."""
        z = self.z_axis()
        x = self.x_axis()
        return np.meshgrid(z, x)

    def r(self):
        """Distance from aperture centre for axial plot."""
        z, x = self.zx_plane()
        return np.sqrt(z**2 + x**2)

    def axial_angle(self):
        """Azimuth(x) angle to point(z, x)."""
        z, x = self.zx_plane()
        return np.arctan2(x, z)

    def p_azimuth(self):
        """Calculate pressure field in the azimuth plane (zx)."""
        arg = self.width_lambda * np.sin(self.axial_angle())
        if self.circular:
            p1 = bpu.jinc(arg)
        else:
            p1 = np.sinc(arg)
        return 1 / self.r() * p1

    def p_elevation(self):
        """Calculate pressure field in the elevation plane (zy)."""
        arg = self.height_lambda * np.sin(self.axial_angle())
        if self.circular:
            return self.p_azimuth()
        else:
            return 1 / self.r() * np.sinc(arg)

    @property
    def db_scale(self):
        """Calculate dB-scale limits from gain and dynamic range."""
        return np.array([-self.db_range, 0]) - self.db_gain

    @property
    def db_ticks(self):
        """Calculate dB-scale limits from gain and dynamic range."""
        db_sep = 6

        vmin, vmax = self.db_scale
        ticks = np.arange(
            db_sep * np.floor(vmin / db_sep),
            db_sep * np.ceil(vmax / db_sep) + db_sep,
            db_sep,
        )
        return ticks

    # === Lateral plane ===================

    def xy_plane(self):
        """Lateral region to plot, plane at fixed axial distance."""
        x = np.linspace(-self.x_max, self.x_max, 201)
        return np.meshgrid(x, x)

    def azimuth_angle_xy(self):
        """Azimuth angles for xy-positions at distance z."""
        x, y = self.xy_plane()
        return np.arctan2(x, self.reference_distance)

    def elevation_angle_xy(self):
        """Elevation angles for xy-positions at distance z."""
        x, y = self.xy_plane()
        return np.arctan2(y, self.reference_distance)

    def radial_angle_xy(self):
        """Angle with z-axis for (xyz)-positions."""
        x, y = self.xy_plane()
        r = np.sqrt(x**2 + y**2)  # Radial distance
        return np.arctan2(r, self.reference_distance)

    def r_xy(self):
        """Radial distances at lateral plane."""
        x, y = self.xy_plane()
        r = np.sqrt(x**2 + y**2)  # Radial distance
        return np.sqrt(r**2 + self.reference_distance**2)

    def p_lateral(self):
        """Calculate lateral amplitude at reference distance."""
        if self.circular:
            arg = self.width_lambda * np.sin(self.radial_angle_xy())
            p = 1 / self.r_xy() * bpu.jinc(arg)
        else:
            arg_az = self.width_lambda * np.sin(self.azimuth_angle_xy())
            arg_el = self.height_lambda * np.sin(self.elevation_angle_xy())

            p_az = np.sinc(arg_az)
            p_el = np.sinc(arg_el)
            p = 1 / self.r_xy() * p_az * p_el

        return p

    # === Commands =============================
    def update_element(self):
        """Update aperture illustration, shape and dimensions."""
        w_mm = self.width * 1e3
        h_mm = self.height * 1e3

        patch_type = patches.Circle if self.circular else patches.Rectangle

        # Redraw shape if changed
        if not isinstance(self.graphs["element"], patch_type):
            self.graphs["element"].remove()

            if self.circular:
                self.graphs["element"] = patches.Circle(
                    (0, 0),
                    w_mm / 2,
                    fill=True,
                    color=COLOR["element"],
                )
            else:
                self.graphs["element"] = patches.Rectangle(
                    (-w_mm / 2, -h_mm / 2),
                    w_mm,
                    h_mm,
                    fill=True,
                    color=COLOR["element"],
                )

            self.axes["element"].add_patch(self.graphs["element"])

        # Update dimensions
        if self.circular:
            self.graphs["element"].set_radius(w_mm / 2)
        else:
            self.graphs["element"].set_xy((-w_mm / 2, -h_mm / 2))
            self.graphs["element"].set_width(w_mm)
            self.graphs["element"].set_height(h_mm)

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
            np.max(np.abs(p_az)),
            np.max(np.abs(p_el)),
        )

        # Axial
        p_axial = p_az if self.azimuth else p_el

        p_db = bpu.db(p_axial, reference=p_max)
        self.graphs["axial"].set_array(p_db.ravel())

        self.graphs["refline"].set_xdata(
            [self.reference_distance, self.reference_distance]
        )

        # Lateral
        p_db = bpu.db(self.p_lateral(), reference=p_max)
        self.graphs["lateral"].set_array(p_db.ravel())
        self.axes["lateral"].set_title(
            f"Lateral plane at {self.reference_distance:.1f} m"
        )

        # Lateral beam profile
        x = self.x_axis()
        z = self.z_axis()
        k_ref = np.argmin(abs(z - self.reference_distance))
        p = p_axial[:, k_ref]
        p_db = bpu.db(p, reference=p_max)
        self.graphs["beamprofile"].set_data(x, p_db)

        # Find reference values
        ref = bpu.Refpoints(x=x, y=p)
        xl, _ = ref.ref_values(y_rel=self.y_lim)  # Beam width limits
        self.dx = xl[1] - xl[0]

        self.x_sidelobe, self.y_sidelobe = ref.sidelobe()
        self.db_sidelobe = bpu.db(self.y_sidelobe, reference=p.max())

        # Update messages
        resulttext = self.update_resulttext()
        self.graphs["text"].txt.set_text(resulttext)

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

        element_max = self.d_max * 1e3 * np.array([-1, 1])
        ax["element"].set(
            xlim=element_max, 
            ylim=element_max,
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

        ax["beamprofile"].set(xlim = lateral_max)

    def update_resulttext(self):
        """Text box for lateral profile results."""
        header = (
            f"Frequency \t $f$ = {self.frequency/1e3:.0f} kHz\n"
            "Wavelength \t"
            rf"$\lambda$ = {self.wavelength*1e3:.1f} mm"
        )

        if self.circular:  # Height dimension omitted
            size_text = (
                f"Diameter \t $D$ = {self.width*1e3:.0f} mm = "
                rf"{self.width_lambda:.1f} $\lambda$"
            )
        else:
            size_text = (
                f"Width \t\t$w$ = {self.width*1e3:.0f} mm = "
                rf"{self.width_lambda:.1f} $\lambda$"
                "\n"
                f"Height \t\t$h$ = {self.height*1e3:.0f} mm = "
                rf"{self.height_lambda:.1f} $\lambda$"
            )

        angle_text = (
            f"Opening angle ({self.lim_text})"
            "\t"
            r"$\theta_0$ = "
            rf"{np.degrees(self.opening_angle):.1f}$^\circ$"
        )

        distance_text = (
            "Rayleigh distance \t  $z_R$ = " f"{self.rayleigh_distance:.2f} m"
        )
        beamwidth_text = (
            f"Beam width ({self.lim_text}) "
            "\t"
            r" $D_z$ = "
            f"{self.dx:.2f} m"
        )

        if np.isnan(self.x_sidelobe):
            sidelobe_text = ""
        else:
            sidelobe_text = (
                "Highest sidelobe "
                "\t"
                f" $x$ = {abs(self.x_sidelobe):.2f} m, "
                f"{self.db_sidelobe:.1f} dB"
            )

        result_text = (
            header
            + "\n"
            + size_text
            + "\n"
            + "\n"
            + distance_text
            + "\n"
            + angle_text
            + "\n"
            + beamwidth_text
            + "\n"
            + sidelobe_text
        )

        return result_text

    def interact(
        self,
        circular=None,
        azimuth=None,
        freq_khz=None,
        width_mm=None,
        height_mm=None,
        distance=None,
        db_range=None,
        db_gain=None,
    ):
        """
        Scale inputs and  display the resulting response.

        For interactive operation with dimensions in mm and frequency in kHz.
          Existing values are retained if a parameter is omitted.

        Parameters
            ----------
            circular: bool, optional
                Circular (True) or rectangular (False) aperture
            azimuth: bool, optional
                Azimuth (True) or elevation (False) orientation
            freq_khz: float, optional
                Frequency in kHz
            width_mm: float, optional
                Element width (azimuth, x) in mm
            height_mm: float, optional
                Element height (elevation, y) in mm
            distance: float, optional
                Reference depth in m
            db_range: float
                Range on dB-axes
            db_gain: float
                Maximum on dB-axes
        """
        if circular is not None:
            self.circular = circular

        if azimuth is not None:
            self.azimuth = azimuth

        if freq_khz is not None:
            self.frequency = float(freq_khz) * 1e3

        if width_mm is not None:
            self.width = float(width_mm) * 1e-3

        if height_mm is not None:
            self.height = float(height_mm) * 1e-3

        if distance is not None:
            self.distance = float(distance)

        if db_range is not None:
            self.db_range = db_range

        if db_gain is not None:
            self.db_gain = db_gain

        if any(
            v is not None
            for v in (
                circular,
                azimuth,
                width_mm,
                height_mm,
            )
        ):
            self.update_element()

        if any(
            v is not None
            for v in (
                circular,
                azimuth,
                freq_khz,
                width_mm,
                height_mm,
                distance,
            )
        ):
            self.update_values()

        if any(v is not None for v in (db_range, db_gain)):
            self.update_intensity_scale()

    # === Non-public methods ==========================================
    # Graphs and results

    def _create_element(self, ax):
        """Create aperture patch and return handle."""

        ax.set(
            title="Transducer shape",
            facecolor=COLOR["element_background"],
            box_aspect=1,
            xlabel="Azimuth [mm]",
            ylabel="Elevation [mm]",
        )

        if self.circular:
            patch = patches.Circle(
                (0, 0),
                radius=0,
                fill=True,
                color=COLOR["element"],
            )
        else:
            patch = patches.Rectangle(
                (0, 0),
                width=0,
                height=0,
                fill=True,
                color=COLOR["element"],
            )

        ax.add_patch(patch)
        return patch

    def _create_resulttextbox(self, ax):
        """Create and attach a formatted results text box to an Axes.

        The text box is anchored to an axis and remains fixed relative to
        the axes if the figure is resized.
        """
        ax.axis("off")
<<<<<<< HEAD
== == == =
    def _initialise_graphs(self):
        """Initialise result graphs."""
        plt.close('all')

        fig, axes = plt.subplot_mosaic(
            [
                ['element', 'axial', 'axial'],
                ['element', 'axial', 'axial'],
                ['element', 'axial', 'axial'],
                ['text', 'lateral', 'beamprofile'],
                ['text', 'lateral', 'beamprofile'],
                ['logo', 'lateral', 'beamprofile'],
            ],
            figsize=[10, 5],
            layout='constrained',
            num='Single Element Beamprofile',
        )
        graphs = {}
        # bpu.add_logo(fig)
<<<<<<< Updated upstream
>>>>>> > Stashed changes
=======
>>>>>>> Stashed changes
=======
>>>>>>> parent of 4bbe23e (Update single_element_beam_pattern.py)

        # Create empty anchored text box
        at = AnchoredText(
            " ",
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
        """Load logo file and place in specified axis."""
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
        """Create axis for axial intensity plot."""

        ax.set(
            aspect="equal",
            xlabel="Depth (z) [m]",
            ylabel="Lateral position (Azimuth or Elevation) [m]",
            facecolor=COLOR["intensity_background"],
            
        )

        x_coords = self.z_axis()
        y_coords = self.x_axis()
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
        """Create azimuth and elevation reference lines.

        Returns
        -------
        tuple
            (azimuth_lines, elevation_lines)
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

    def _create_lateral_plot(self, ax):
        """Create axis for lateral intensity plots."""
        # Lateral intensity plot
        ax.set(box_aspect=1, xlabel="Azimuth [m]", ylabel="Elevation [m]")

        x_coords = self.xy_plane()[0][0, :]
        y_coords = self.xy_plane()[1][:, 0]
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

    def _create_beamprofile_plot(self, ax):
        """Create axis for beam profile graphs."""
        ax.set(
            xlabel="Distance [m]",
            ylabel="Power [dB re. max]",
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
        plt.close("all")

        fig, axes = plt.subplot_mosaic(
            [
                ["element", "axial", "axial"],
                ["element", "axial", "axial"],
                ["element", "axial", "axial"],
                ["text", "lateral", "beamprofile"],
                ["text", "lateral", "beamprofile"],
                ["logo", "lateral", "beamprofile"],
            ],
            figsize=(12, 6),
            layout="constrained",
            num="Single Element Beamprofile",
        )

        graphs = {}
        self._create_logo(axes["logo"])
        graphs["element"] = self._create_element(axes["element"])
        graphs["text"] = self._create_resulttextbox(axes["text"])
        graphs["axial"] = self._create_axial_plot(axes["axial"])
        graphs["lateral"] = self._create_lateral_plot(axes["lateral"])
        graphs["beamprofile"] = self._create_beamprofile_plot(
            axes["beamprofile"]
        )

        graphs["azimuth"], graphs["elevation"] = (
            self._create_orientation_lines((axes["element"], axes["lateral"]))
        )

        # Reference line for distance
        graphs["refline"] = axes["axial"].axvline(
            x=self.reference_distance,
            **LINE["orientation"],
        )

        # Colorbar for intensity plots
        graphs["colorbar"] = fig.colorbar(
            graphs["axial"],
            ax=axes["axial"],
            label = "dB re. max"
        )
        
        return fig, axes, graphs

    # Interactive widgets
    def _create_widgets(self):
        """Create widgets for interactive operation."""
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
            description="Width (Diameter) [mm]",
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

        widget = {
            "circular": shape_widget,
            "azimuth": orientation_widget,
            "db_range": db_range_widget,
            "db_gain": db_gain_widget,
            "frequency": frequency_widget,
            "width": width_widget,
            "height": height_widget,
            "distance": distance_widget,
        }

        return WidgetLayout(widget_layout, widget)
