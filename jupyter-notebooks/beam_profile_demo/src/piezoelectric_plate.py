from math import pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

FIGURE_NAME = "Piezoelectric Plate Demo"


class Voltmeter:
    """Define and draw a voltmeter."""

    def __init__(self, radius=1.0, x=2.0, y=0.0, value=0.0):
        self.radius = radius
        self.x = x
        self.y = y
        self.value = np.clip(value, -1.0, 1.0)
        self.max_angle = 3 * pi / 4
        self.pointer = None

    @property
    def pointer_angle(self):
        return pi / 2 - self.value * self.max_angle

    @property
    def top(self):
        return self.y + self.radius

    @property
    def centre(self):
        return self.x, self.y

    @property
    def bottom(self):
        return self.y - self.radius

    def draw(self, ax):
        """Draw voltmeter with pointer."""
        self.ax = ax

        circle = plt.Circle(
            self.centre,
            self.radius,
            color="#F4F1E1",  # "#F4F1E1", "#EDE7D1","#F0EAD6"
            ec="black",
            lw=1.0,
            zorder=3,
        )

        ax.add_patch(circle)

        scale_angle = np.linspace(
            pi / 2 + self.max_angle, pi / 2 - self.max_angle, 31
        )

        scale_radius = 0.80 * self.radius

        scale_x = self.x + scale_radius * np.cos(scale_angle)
        scale_y = self.y + scale_radius * np.sin(scale_angle)
        ax.plot(scale_x, scale_y, color="black", lw=1.0, zorder=4)

        tick_length = 0.16 * self.radius
        for rad in scale_angle:
            tx_start = self.x + (scale_radius - tick_length) * np.cos(rad)
            tx_end = self.x + scale_radius * np.cos(rad)
            ty_start = self.y + (scale_radius - tick_length) * np.sin(rad)
            ty_end = self.y + scale_radius * np.sin(rad)

            ax.plot(
                [tx_start, tx_end],
                [ty_start, ty_end],
                color="black",
                lw=0.7,
                zorder=4,
            )

        self.pointer = FancyArrowPatch(
            self.centre,
            self.centre,
            arrowstyle="-|>",
            color="crimson",
            lw=3,
            mutation_scale=15,
            zorder=5,
        )

        ax.add_patch(self.pointer)

        ax.plot(self.x, self.y, "o", color="black", markersize=8, zorder=6)

        self.update_pointer(self.value)

    def update_pointer(self, value):
        if self.pointer is None:
            return

        self.value = value
        pointer_length = 0.8 * self.radius
        x = self.x + pointer_length * np.cos(self.pointer_angle)
        y = self.y + pointer_length * np.sin(self.pointer_angle)

        self.pointer.set_positions(
            self.centre,
            (x, y),
        )


class Plate:
    """Define and draw a piezoelectric plate."""

    def __init__(self, x=6.0, y=0.0, width=4.0, thickness=1.0):
        self.x = x
        self.y = y
        self.width = width
        self.thickness = thickness

    @property
    def center_x(self):
        return self.x + self.width / 2

    @property
    def top(self):
        return self.y + self.thickness / 2

    @property
    def bottom(self):
        return self.y - self.thickness / 2

    def draw(self, ax):
        self.plate = patches.Rectangle(
            (self.x, -self.thickness / 2),
            self.width,
            self.thickness,
            facecolor="#8B8580",
            lw=2,
            zorder=3,
        )

        ax.add_patch(self.plate)

        self.electrodes = []
        for y in (self.top, self.bottom):
            (line,) = ax.plot(
                [self.x, self.x + self.width],
                [y, y],
                linewidth=8.0,
                color="#CD7F32",  # "#8B8580", "#CD7F32", "#C97A40"
            )
            self.electrodes.append(line)

    def update_thickness(self, thickness):

        self.thickness = thickness

        for electrode, y in zip(self.electrodes, (self.top, self.bottom)):
            electrode.set_ydata([y, y])

        self.plate.set_y(self.bottom)
        self.plate.set_height(self.thickness)


# Draw connections
class Connections:

    def draw(self, ax, voltmeter):
        """Draw conections between voltmeter and plate."""
        dx = voltmeter.radius * 1.5
        dy = 0.5
        self.wire_x = [
            voltmeter.x,
            voltmeter.x,
            voltmeter.x + dx,
            voltmeter.x + dx,
            voltmeter.x + 2 * dx,
        ]

        upper = (
            voltmeter.top,
            voltmeter.top + dy,
            voltmeter.top - dy,
        )
        lower = (
            voltmeter.bottom,
            voltmeter.bottom - dy,
            voltmeter.bottom + dy,
        )

        self.wire_y = []

        for start_y, mid_y, end_y in (upper, lower):
            self.wire_y.append([start_y, mid_y, mid_y, end_y, end_y])

        for y in self.wire_y:
            ax.plot(self.wire_x, y, color="black", lw=1)


class Arrow:
    def __init__(self, x, y, length=1, down=False):
        self.x = x
        self.y = y
        self.length = length
        self.down = down

    def draw(self):
        if self.down:
            y_length = self.length
        else:
            y_length = -self.length

        ax.annotate(
            "",
            xy=(self.x, self.y),
            xytext=(self.x, self.y + y_length),
            arrowprops=dict(
                facecolor="darkred",
                edgecolor="darkred",
                width=6,
                headwidth=20,
                shrink=0.05,
            ),
            zorder=4,
        )


# --- Main program ------------------------------------------------
plt.close("Piezoelectric Plate Demo")
drawing_row = ["drawing"] * 4

fig, axes = plt.subplot_mosaic(
    [
        ["text"] + drawing_row,
        ["logo"] + drawing_row,
    ],
    figsize=(12, 8),
    layout="constrained",
    num=FIGURE_NAME,
)
ax = axes["drawing"]

# --- Draw objects --------------------------------------------------
voltmeter = Voltmeter(value=0.0)
voltmeter.draw(ax)

plate = Plate()
plate.draw(ax)

connection = Connections()
connection.draw(ax, voltmeter)

top_arrow = Arrow(plate.center_x, plate.top, length=1, down=True)
top_arrow.draw()

bottom_arrow = Arrow(plate.center_x, plate.bottom, length=1, down=False)
bottom_arrow.draw()

# Format axes -------------------------------------------------------
ax.set_xlim(0, 12)
ax.set_ylim(-4, 4)
ax.axis("off")
ax.axis("equal")

plt.tight_layout()
plt.show()
