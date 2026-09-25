from math import pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import ipywidgets as widgets
from pathlib import Path
import matplotlib.image as mpimg

FIGURE_NAME = "Piezoelectric Plate Demo"
LOGOFILE = "usn-logo-purple.png"


class Voltmeter:
    """
    Define and draw a voltmeter.

    The arrow position can be changes after the voltmeter is drawn
    """

    def __init__(self, x=2.0, y=0.0, width=2.4, height=1.0, value=0.0):
        """
        Draw the voltmeter with specified size, position , and value

        Parameters
        ----------
        x : float , optional
            Voltmeter x-position. The default is 2.0.
        y : float, optional
            Voltmeter y-position. The default is 0.0.
        value : float, optional
            Voltmeter reading, full scale is +/-1.0. The default is 0.0.

        Returns
        -------
        None
        """
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.value = np.clip(value, -1.0, 1.0)
        self.max_angle = 0.20 * pi  # Angle corresponding to value=1.0
        self.pointer = None

    @property
    def pointer_angle(self):
        """Convert value to angle in radians."""
        return pi / 2 - self.value * self.max_angle

    @property
    def top(self):
        """y-coordinate of voltmeter top."""
        return self.y + self.height / 2

    @property
    def bottom(self):
        """y-coordinate of voltmeter bottom."""
        return self.y - self.height / 2

    @property
    def left(self):
        """x-coordinate of voltmeter left."""
        return self.x - self.width / 2

    @property
    def right(self):
        """x-coordinate of voltmeter right."""
        return self.x + self.width / 2

    @property
    def centre(self):
        """(x,y) coordinates of voltmeter centre."""
        return self.x, self.y

    def draw(self, ax):
        """
        Draw voltmeter with pointer.


        Parameters
        ----------
        ax : Matplotlib axis
            Axis to draw voltmeter in.

        Returns
        -------
        None
        """
        rectangle = patches.Rectangle(
            xy=(self.x - self.width / 2, self.y - self.height / 2),
            width=self.width,
            height=self.height,
            color="#FAF8F0",  # "#F2F0E6", "#FAF8F0","#E9EEF2"
            ec="darkgrey",
            linewidth=1.5,
            zorder=3,
        )

        ax.add_patch(rectangle)

        # Draw scale
        n_ticks = 15
        scale_angle = np.linspace(
            pi / 2 + self.max_angle, pi / 2 - self.max_angle, n_ticks
        )

        scale_centre_x = self.x
        scale_centre_y = self.y - 1.2 * self.height
        self.scale_centre = (scale_centre_x, scale_centre_y)

        scale_radius = 1.5 * self.height
        scale_x = scale_centre_x + scale_radius * np.cos(scale_angle)
        scale_y = scale_centre_y + scale_radius * np.sin(scale_angle)
        ax.plot(
            scale_x,
            scale_y,
            color="black",
            linewidth=1.0,
            zorder=4,
        )

        indices = [0, (n_ticks - 1) // 2, n_ticks - 1]
        labels = ["-1", "0", "+1"]
        positions = ["right", "center", "left"]
        for label, idx, pos in zip(labels, indices, positions):
            ax.text(
                scale_x[idx],
                scale_y[idx],
                label,
                ha=pos,
                va="bottom",
                size="large",
            )

        # Draw scale markers
        tick_length = 0.08
        for rad in scale_angle:
            tx_start = scale_centre_x + (scale_radius - tick_length) * np.cos(
                rad
            )
            tx_end = scale_centre_x + scale_radius * np.cos(rad)
            ty_start = scale_centre_y + (scale_radius - tick_length) * np.sin(
                rad
            )
            ty_end = scale_centre_y + scale_radius * np.sin(rad)

            ax.plot(
                [tx_start, tx_end],
                [ty_start, ty_end],
                color="black",
                linewidth=0.7,
                zorder=4,
            )

        # Draw pointer
        self.pointer = FancyArrowPatch(
            (scale_centre_x, scale_centre_y),
            (scale_centre_x, scale_centre_y),
            arrowstyle="->",
            color="crimson",
            linewidth=4,
            mutation_scale=20,
            zorder=5,
        )
        ax.add_patch(self.pointer)

        # Text
        ax.text(
            self.x,
            self.bottom + self.height * 0.1,
            "Volts",
            ha="center",
            va="bottom",
            size="large",
            zorder=6,
        )

        self.update(self.value)

    def update(self, value):
        """
        Update voltmeter pointer to show specified value

        Parameters
        ----------
        value : float
            Voltmeter reading.

        Returns
        -------
        None
        """
        if self.pointer is None:
            return

        self.value = np.clip(value, -1.0, 1.0)
        pointer_length = 1.5 * self.height

        x0, y0 = self.scale_centre
        pointer_start = 0.70 * pointer_length
        cp = np.cos(self.pointer_angle)
        sp = np.sin(self.pointer_angle)

        x_start = x0 + pointer_start * cp
        y_start = y0 + pointer_start * sp

        x_end = x0 + pointer_length * cp
        y_end = y0 + pointer_length * sp

        self.pointer.set_positions(
            (x_start, y_start),
            (x_end, y_end),
        )


class Plate:
    """Define and draw a piezoelectric plate."""

    def __init__(self, x=6.0, y=0.0, width=3.0, thickness=0.5):
        """
        Draw the piezoelectric plate with specified size and position.

        Parameters
        ----------
        x : float, optional
            Center of plate, x-coordinate. The default is 6.0.
        y : float, optional
            Center of plate, y-coordinate. The default is 0.0.
        width : float, optional
            Width of plate, x-coordinate units. The default is 4.0.
        thickness : float, optional
            Thickness plate, y-coordinate units. The default is 1.0.

        Returns
        -------
        None
        """
        self.x = x
        self.y = y
        self.width = width
        self.thickness = thickness
        self.expansion = 0

    @property
    def current_thickness(self):
        return (1 + self.expansion) * self.thickness

    @property
    def left(self):
        return self.x - self.width / 2

    @property
    def right(self):
        return self.x + self.width / 2

    @property
    def top(self):
        return self.y + self.current_thickness / 2

    @property
    def bottom(self):
        return self.y - self.current_thickness / 2

    def draw(self, ax):

        # Rectangular plate
        self.plate = patches.Rectangle(
            (self.left, self.bottom),
            self.width,
            self.current_thickness,
            facecolor="#8B8580",
            linewidth=0,
            zorder=3,
        )

        ax.add_patch(self.plate)

        # Electrodes
        self.electrodes = []
        for y in (self.top, self.bottom):
            (line,) = ax.plot(
                [self.left, self.left + self.width],
                [y, y],
                linewidth=6,
                color="#CD7F32",  # "#8B8580", "#CD7F32", "#C97A40"
                zorder=4,
            )
            self.electrodes.append(line)

    def update(self, expansion):
        """
                Update plate thickness to specified value
        .
                Parameters
                ----------
                thickness : float
                    Plate thickness in y-coordinates

                Returns
                -------
                None
        """
        self.expansion = np.clip(expansion, -0.9, 3.0)

        for electrode, y in zip(self.electrodes, (self.top, self.bottom)):
            electrode.set_ydata([y, y])

        self.plate.set_y(self.bottom)
        self.plate.set_height(self.top - self.bottom)


# Draw connections
class Wires:
    """Wires from voltmeter."""

    @property
    def endpoints(self):
        upper = self.x[-1], self.y[0][-1]
        lower = self.x[-1], self.y[1][-1]
        return (upper, lower)

    def draw(self, ax, voltmeter):
        """Draw connections to voltmeter."""
        self.x = [
            voltmeter.right,
            voltmeter.right + 0.3 * voltmeter.width,
        ]

        upper = (
            voltmeter.y + voltmeter.height / 4,
            voltmeter.y + voltmeter.height / 4,
        )
        lower = (
            voltmeter.y - voltmeter.height / 4,
            voltmeter.y - voltmeter.height / 4,
        )

        self.y = [upper, lower]

        for y in self.y:
            ax.plot(self.x, y, color="black", lw=1)


class Connection:
    """Connection between voltmeter wires and piezoelectric plate."""

    def __init__(self):
        self.x = []
        self.y = []
        self.lines = []

    def draw(self, ax, endpoints, plate):

        upper, lower = endpoints
        self.x = [upper[0], plate.left]
        y_top = [upper[1], plate.top]
        y_bottom = [lower[1], plate.bottom]

        self.lines = ax.plot(self.x, y_top, self.x, y_bottom, color="black")
        self.y = [y_top, y_bottom]

    def update(self, endpoints):
        """Update last connection points y-values."""
        for line, y, new_y in zip(self.lines, self.y, endpoints):
            y[-1] = new_y
            line.set_ydata(y)


class Forces:
    """Arrows illustrating the the forces on top and bottom of the plate."""

    def __init__(self, length=0.3):
        self.length = length
        self.arrows = ([], [])

    def _x_pos(self, plate):
        """Determine lateral positions (x) of force/pressure arrows."""
        n_arrows = 6
        x_offset = plate.x - plate.width / 2
        x = np.arange(1, n_arrows) / n_arrows * plate.width + x_offset

        return x

    def draw(self, ax, plate, value):

        force_style = {
            "arrowstyle": "-|>",
            "color": "#1F618D",  # "#1C2833" "#0B5345" "#641E16" "#1F618D"
            "linewidth": 2,
            "mutation_scale": 28,
            "zorder": 5,
        }

        x_pos = self._x_pos(plate)

        top_arrows = []
        bottom_arrows = []
        for x in x_pos:
            top_arrows.append(
                FancyArrowPatch(
                    (x, plate.top),
                    (x, plate.top + self.length),
                    **force_style,
                )
            )

            bottom_arrows.append(
                FancyArrowPatch(
                    (x, plate.bottom - self.length),
                    (x, plate.bottom),
                    **force_style,
                )
            )

        for arrow in top_arrows:
            ax.add_patch(arrow)
        for arrow in bottom_arrows:
            ax.add_patch(arrow)

        self.arrows = (top_arrows, bottom_arrows)

        self.update(plate, value)

    def update(self, plate, value):
        """
        Update arrows illustrating force

        Parameters
        ----------
        value : float
            Force value.

        Returns
        -------
        None
        """
        length = self.length

        x_pos = self._x_pos(plate)
        end_ys = (plate.top, plate.bottom)
        start_ys = (plate.top + length, plate.bottom - length)

        for arrow_group, end_y, start_y in zip(self.arrows, end_ys, start_ys):
            for x, arrow in zip(x_pos, arrow_group):

                if arrow is None:
                    return

                end = (x, end_y)
                start = (x, start_y)

                if value < 0:
                    start, end = end, start
                if value == 0:
                    end = start

                arrow.set_positions(start, end)


class PiezoelectricPlate:
    """Demonstration of piezoelectric effect, plate connected to voltmeter."""

    def __init__(self, create_widgets=False):
        self.value = 0.0

        plt.close(FIGURE_NAME)

        if create_widgets:
            self.widget_layout, self.widgets = self._create_widgets()

        self.fig, self.axes = self._initialise_graph()

        ax = self.axes["drawing"]

        self.voltmeter = Voltmeter(value=self.value)
        self.voltmeter.draw(ax)

        self.plate = Plate()
        self.plate.draw(ax)

        self.wires = Wires()
        self.wires.draw(ax, self.voltmeter)

        self.connection = Connection()
        self.connection.draw(ax, self.wires.endpoints, self.plate)

        self.forces = Forces()
        self.forces.draw(ax, self.plate, self.value)

    def _initialise_graph(self):
        n_col = 4
        drawing_row = ["drawing"] * n_col

        fig, axes = plt.subplot_mosaic(
            [
                drawing_row,
                drawing_row,
                drawing_row,
                ["logo"] + ["."] * (n_col - 1),
            ],
            figsize=(12, 4),
            layout="tight",
            num=FIGURE_NAME,
        )

        self._create_logo(axes["logo"])

        ax = axes["drawing"]

        ax.axis("off")
        ax.set_xlim(0.5, 8.7)
        ax.set_ylim(-0.7, 0.7)
        ax.set_aspect("equal", adjustable="box")

        return fig, axes

    def change_values(self, value):
        """Update system to new value."""
        value = np.clip(value, -1, 1)
        self.voltmeter.update(value)

        self.expansion = -0.5 * value
        self.plate.update(self.expansion)
        self.connection.update([self.plate.top, self.plate.bottom])

        self.forces.update(self.plate, value)

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

    def _force_change_callback(self, change):
        self.value = float(change["new"])
        self.change_values(self.value)

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
        title = "Illustration of the Piezoelectric Effect"
        title_widget = widgets.Label(
            title,
            style=dict(font_weight="bold"),
        )

        layout = {
            "continuous_update": True,
            "layout": widgets.Layout(width="50%"),
            "style": {"description_width": "10%"},
        }

        force_widget = widgets.FloatSlider(
            value=0,
            min=-1.0,
            max=1.0,
            step=0.05,
            readout_format=".2f",
            description="Force",
            **layout,
        )
        force_widget.observe(
            self._force_change_callback,
            names="value",
        )

        widget_layout = widgets.VBox([title_widget, force_widget])

        widget_list = {"force": force_widget}

        return widget_layout, widget_list
