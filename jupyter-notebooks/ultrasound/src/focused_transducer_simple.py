"""Run this to import libraries and define internal functions."""

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ipywidgets as widgets
from pathlib import Path

COLOR = {
    "aperture": "#A63D1F",  # "#B64926"  "#A63D1F" "#B35A1F" "#8C2D19"
    "baffle": "#B0B3B8",
    "text_face": "#F0FBFF",  # "#E6F3F7", " # "#F0FBFF", "#EAF7FA"
}

LINEFORMAT = {
    "aperture": {
        "color": COLOR["aperture"],
        "linestyle": "solid",
        "linewidth": 2.0,
    },
    "focus": {"color": "C1", "linestyle": "solid"},
    "rayleigh": {"color": "red", "linestyle": "dashed"},
    "helper": {"color": "darkgrey", "linestyle": "dashed"},
    "beam": {"color": "C0", "linestyle": "solid"},
}

FILL = {"focalzone": {"color": "C1", "alpha": 0.7}}

LOGOFILE = "usn-logo-purple.png"
FIGURE_NAME = "Transducer Focusing"


class Transducer:
    """Define, calculate, and display transducer beam profile."""

    def __init__(self, create_widgets=False):

        # Transducer definition
        self.diameter = 20e-3  # m   Element diameter
        self.frequency = 3e6  # Hz  Ultrasound frequency
        self.focal_length = 50e-3  # m  Focal length

        self.c = 1540  # m/s Speed of sound in load medium

        # Display scale
        self.x_max = 50e-3
        self.z_max = 200e-3
        self.z_min = -5e-3

        self.z_axis = np.linspace(0, self.z_max, 300)

        # Initialisation
        self.fig, self.axes, self.graphs = self._initialise_graphs()
        self.update_lines()
        self.update_resulttext()

        if create_widgets:
            self.widget_layout, self.widgets = self._create_widgets()

    # === Calculated parameters ===========================
    @property
    def radius(self):
        """Calculate transducer radius."""
        return self.diameter / 2

    @property
    def wavelength(self):
        """Calculate acoustic wavelength."""
        return self.c / self.frequency

    @property
    def diameter_wavelength(self):
        """Calculate aperture width relative to wavelength."""
        return self.diameter / self.wavelength

    @property
    def opening_angle(self):
        """Calculate opening angle from theory, two-sided, -12 dB."""
        return 2 * self.wavelength / self.diameter

    @property
    def f_number(self):
        """Calculate numeric aperture (f-number)."""
        return self.focal_length / self.diameter

    @property
    def rayleigh_distance(self):
        """Calculate Rayleigh distance, far-field limit."""
        return self.diameter**2 / (2 * self.wavelength)

    @property
    def beamwidth(self):
        """Estimate beam width from opening angle."""
        return self.opening_angle * self.focal_length

    @property
    def focalzone_depth(self):
        """Find depth limits of the focal zone."""
        c = self.opening_angle * self.f_number
        if c >= 1:
            return np.nan, np.nan

        z1 = self.focal_length / (1 + c)
        z2 = self.focal_length / (1 - c)

        return z1, z2

    @property
    def focalzone(self):
        """
        Find limits of the focal zone.
        Returns results as corners of ploygon

        """
        z1, z2 = self.focalzone_depth
        
        if np.isnan(z1) or np.isnan(z2):
            return np.full((4, 2), np.nan)

        x1 = z1 * self.opening_angle / 2
        x2 = z2 * self.opening_angle / 2

        z = np.array([z1, z2, z2, z1])
        x = np.array([x1, x2, -x2, -x1])

        return np.column_stack([z, x])

    @property
    def focalzone_length(self):
        """Find length of focal zone."""
        fz = self.focalzone
        z = fz[:, 0]

        return z.max() - z.min()

    @property
    def focalzone_length_approx(self):
        """Calculate approximate length of focal zone."""
        return 4 * self.wavelength * self.f_number**2

    # === Calculated parameters ===================
    def aperture_curve(self):
        """Calculate position of aperture."""
        phi_max = np.arcsin(self.radius / self.focal_length)
        phi = np.linspace(-phi_max, phi_max, 101)

        x = self.focal_length * np.sin(phi)
        z = self.focal_length * (1 - np.cos(phi))

        return z, x

    def diffraction_curve(self):
        """Calculate outer beam profile from diffraction (opening angle)."""
        return self.z_axis * np.tan(self.opening_angle / 2)

    def focusing_curve(self):
        """Calculate outer beam profile from focusing."""
        return self.radius * (1 - self.z_axis / self.focal_length)

    def beam_curve(self):
        """Estimate beam profile from combined opening angle and focusing."""
        return np.maximum(
            abs(self.diffraction_curve()), abs(self.focusing_curve())
        )

    # === Commands =============================
    def update_lines(self):
        """Update graph with beam pattern and guide lines."""

        # Guide lines
        focal_length_mm = self.focal_length * np.ones(2) * 1e3
        self.graphs["focus"].set_xdata(focal_length_mm)

        rayleigh_distance_mm = self.rayleigh_distance * np.ones(2) * 1e3
        self.graphs["rayleigh_distance"].set_xdata(rayleigh_distance_mm)

        diffraction_mm = self.diffraction_curve() * 1e3
        for graph, sign in zip(self.graphs["diffraction"], [-1, 1]):
            graph.set_ydata(sign * diffraction_mm)

        # Beam
        beam_mm = self.beam_curve() * 1e3
        for graph, sign in zip(self.graphs["beam"], [-1, 1]):
            graph.set_ydata(sign * beam_mm)

        # Aperture
        z, x = self.aperture_curve()
        z_mm = z * 1e3
        x_mm = x * 1e3
        self.graphs["aperture"].set_data(z_mm, x_mm)

        z_fill_mm = np.pad(z_mm, (1, 1), "constant", constant_values=0)
        x_fill_mm = np.pad(x_mm, (1, 1), "edge")
        aperture_fill = np.column_stack([z_fill_mm, x_fill_mm])
        self.graphs["aperture_fill"].set_xy(aperture_fill)

        # Focal zone
        focal_zone_mm = self.focalzone * 1e3
        self.graphs["focalzone"].set_xy(focal_zone_mm)

    def update_resulttext(self):
        """Update text box with array parameters."""

        z1, z2 = self.focalzone_depth

        value_lines = [
            f"{self.frequency/1e6:.2f} MHz",
            rf"{self.wavelength*1e6:.0f} $\mu$m",
            f"{self.diameter*1e3:.1f} mm",
            f"{self.focal_length*1e3:.0f} mm",
            f"{self.focal_length/self.diameter:.1f}",
            f"{self.rayleigh_distance*1e3:.0f} mm",
            rf"{np.degrees(self.opening_angle):.1f}$^\circ$",
            f"{self.beamwidth*1e3:.1f} mm",
            f"{z1*1e3:.1f} mm",
            f"{z2*1e3:.1f} mm",
            f"{self.focalzone_length*1e3:.1f} mm",
        ]

        for line_no, value in enumerate(value_lines):
            self.graphs["text"][(line_no, 2)].get_text().set_text(value)

        lambda_symbol = r"$\lambda$"

        self.graphs["text"][(2, 3)].get_text().set_text(
            f"{self.diameter_wavelength:.1f} " + lambda_symbol,
        )

        self.graphs["text"][(7, 3)].get_text().set_text(
            f"{self.beamwidth/self.wavelength:.1f} " + lambda_symbol,
        )

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
            ["Frequency", "$f$", "", ""],
            ["Wavelength", r"$\lambda$", "", ""],
            ["Diameter", "$D$", "", ""],
            ["Focal length", "$F$", "", ""],
            ["F-number", "$FN$", "", ""],
            ["Rayleigh distance ", r"$z_R$", "", ""],
            [
                "Opening angle, -12 dB",
                r"$\theta_{12dB}$",
                "",
                "",
            ],
            ["Beam width", "$D_F$", "", ""],
            ["Focal zone", r"$z_{F1}$", "", ""],
            ["", r"$z_{F2}$", "", ""],
            ["Focal zone length", r"$L_F$", "", ""],
        ]

        table = ax.table(
            cellText=resulttext,
            loc="upper left",
            cellLoc="left",
            colWidths=[0.50, 0.15, 0.20, 0.15],
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

    def _create_beam_plot(self, ax):
        """
        Display beam pattern in graphs.

        Axes are defined so that
        Lateral axis, x -> Plot axis y
        Depth axis z -> Plot axis x
        """
        x_max_mm = self.x_max * 1e3
        z_lim_mm = np.array([self.z_min, self.z_max]) * 1e3

        ax.set(
            aspect="equal",
            xlabel="Depth (z) [mm]",
            ylabel="Lateral (x) [mm]",
            ylim=(-x_max_mm, x_max_mm),
            xlim=z_lim_mm,
        )
        ax.grid(visible="True", which="both")

        # Baffle, static
        ax.axvspan(xmin=z_lim_mm[0], xmax=0, color=COLOR["baffle"])

        graphs = {}

        # Aperture
        (graphs["aperture"],) = ax.plot([], [], **LINEFORMAT["aperture"])
        (graphs["aperture_fill"],) = ax.fill([], [], color=COLOR["aperture"])

        # Marker lines
        graphs["focus"] = ax.axvline(x=10, **LINEFORMAT["focus"])
        graphs["rayleigh_distance"] = ax.axvline(
            x=20, **LINEFORMAT["rayleigh"]
        )

        # Result lines. Depth-values (x axis) are fixed
        z_axis_mm = self.z_axis * 1e3
        dummy_data = np.zeros_like(z_axis_mm)
        graphs["diffraction"] = ax.plot(
            z_axis_mm,
            dummy_data,
            z_axis_mm,
            dummy_data,
            **LINEFORMAT["helper"],
        )
        graphs["beam"] = ax.plot(
            z_axis_mm,
            dummy_data,
            z_axis_mm,
            dummy_data,
            **LINEFORMAT["beam"],
        )

        # Mark focal zone
        (graphs["focalzone"],) = ax.fill([], [], **FILL["focalzone"])

        return graphs

    def _initialise_graphs(self):
        """Initialise result graphs."""
        plt.close(FIGURE_NAME)

        beam_row = ["beam"] * 3
        fig, axes = plt.subplot_mosaic(
            [
                ["text"] + beam_row,
                ["text"] + beam_row,
                ["text"] + beam_row,
                ["logo"] + beam_row,
            ],
            figsize=(14, 6),
            layout="constrained",
            num=FIGURE_NAME,
        )

        graphs = self._create_beam_plot(axes["beam"])
        graphs["text"] = self._create_resulttextbox(axes["text"])
        self._create_logo(axes["logo"])

        return fig, axes, graphs

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

    # Callback functions
    def _refresh(self):
        self.update_lines()
        self.update_resulttext()
        self.fig.canvas.draw_idle()

    def _frequency_change_callback(self, change):
        self.frequency = float(change["new"]) * 1e6
        self._refresh()

    def _diameter_change_callback(self, change):
        self.diameter = float(change["new"]) / 1e3
        self._refresh()

    def _focal_length_change_callback(self, change):
        self.focal_length = float(change["new"]) / 1e3
        self._refresh()

    # Interactive widgets
    def _create_widgets(self):
        """Create widgets for interactive operation."""
        title = "Beam-profile from Focused Transducer. Simple Estimate"
        title_widget = widgets.Label(title, style=dict(font_weight="bold"))

        slider_layout = {
            "continuous_update": True,
            "layout": widgets.Layout(width="95%"),
            "style": {"description_width": "30%"},
        }

        right_width = "95%"

        frequency_widget = widgets.FloatSlider(
            min=0.1,
            max=10.0,
            value=self.frequency / 1e6,
            step=0.1,
            readout_format="3.1f",
            description="Frequency [MHz]",
            **slider_layout,
        )
        frequency_widget.observe(
            self._frequency_change_callback,
            names="value",
        )

        diameter_widget = widgets.FloatSlider(
            min=1,
            max=50,
            value=self.diameter * 1e3,
            step=1,
            readout_format=".0f",
            description="Diameter [mm]",
            **slider_layout,
        )
        diameter_widget.observe(
            self._diameter_change_callback,
            names="value",
        )

        focal_length_widget = widgets.FloatSlider(
            min=1,
            max=150,
            value=self.focal_length * 1e3,
            step=1,
            readout_format=".0f",
            description="Focal length [mm]",
            **slider_layout,
        )
        focal_length_widget.observe(
            self._focal_length_change_callback,
            names="value",
        )

        widget_layout = widgets.VBox(
            [frequency_widget, diameter_widget, focal_length_widget],
            layout=widgets.Layout(width=right_width),
        )

        widget_layout = widgets.VBox([title_widget, widget_layout])

        widget = {
            "diameter": diameter_widget,
            "frequency": frequency_widget,
            "focal_length": focal_length_widget,
        }

        return widget_layout, widget
