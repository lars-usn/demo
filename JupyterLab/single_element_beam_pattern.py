"""Run this to import libraries and define internal functions."""

# Libraries
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches

THETA_MAX = 90      # Deg  Angular extent to plot
Z_REF = 200e-3      # m    Axial distance to calculate lateral intensity
X_MAX = 60e-3       # m    Lateral extent of plot region
SOUND_SPEED = 1540  # m/s  Speed of sound in medium


# %% Classes
class Aperture:
    """Transducer aperture definition.

    Atteributes
    -----------
    shape       String  "Rectangular" or "Circular"
    width       Float   Element width [m]
    height      Float   Element height [m]
    frequency   Float   All calculations are for a single frequency (CW) [Hz]
    c           Float   Speed of sound [m/s]

    Methods
    -------
    wavelength  Float   Acoustic wavelength [m]
    """

    shape = "Rectangular"
    width = 10e-3   # m   Element width or diameter
    height = 10e-3   # m   Element width or diameter
    frequency = 1e6    # Hz  Ultrasound frequency

    def wavelength(self):
        """Calculate acoustic wavelenght."""
        return SOUND_SPEED/self.frequency


class PlotRegion:
    """Define region for lateral (xy) beamprofile plots.

    Attributes
    ----------
    theta_max   Float   Max. angle to calculate [Degree]
    x_max       Float   Max. azimoth position calculate [m]
    z_ref       Float   Axial distance for lateral plots [m]

    Methods
    -------
    theta    Float   Aximut angles to calculate [Degree]
    phi      Float   Elavation angles to calculate [Degree]
    xy       Float   Lateral region (xy-plane) to plot [m]
    theta_xy Float   Azimuth angles for xy-positions at distance z_ref [Deg]
    phi_xy   Float   Elevation angles for xy-positions at distance z_ref [Deg]
    theta_r  Float  Angle with z-axis for (xyz)-positions [Deg]
    """

    def __init__(self, theta_max=THETA_MAX, x_max=X_MAX, z_ref=Z_REF):
        """Initialise input variables."""
        self.theta_max = np.radians(theta_max)
        self.x_max = x_max
        self.z = z_ref

    def theta(self):
        """Azimuth (x) angle ."""
        return np.linspace(-self.theta_max, self.theta_max, 301)

    def phi(self):
        """Elevation (y) angle."""
        return np.linspace(-self.theta_max, self.theta_max, 301)

    def xy(self):
        """Lateral region to plot. Plane at fixed distance along axis."""
        pts = np.linspace(-self.x_max, self.x_max, 101)
        return np.meshgrid(pts, pts)

    def theta_xy(self):
        """Azimuth angles for xy-positions at fixed distance z."""
        return np.arctan2(self.xy()[0], self.z)

    def phi_xy(self):
        """Elevation angles for xy-positions at fixed distance z."""
        return np.arctan2(self.xy()[1], self.z)

    def theta_r(self):
        """Angle with z-axis for (xyz)-positions."""
        return np.arctan2(np.sqrt(self.xy()[0]**2+self.xy()[1]**2), self.z)


class Directivity:
    """Directivity patterns for azimuth, elevation, and a lateral plane.

    Attributes
    ----------
    theta   Float   Directivity as function of azimuth angle
    phi     Float   Directivity as function of elavation angle
    xy      Float   Directivoty in xy-plane at reference distance
    """

    theta = np.zeros(1001, dtype=float)
    phi = np.zeros(1001, dtype=float)
    xy = np.zeros((101, 101), dtype=float)


# %% Internal functions

# Mathematical functions
def jinc(x):
    """jinc-function, Bessel-version of sinc, 2J(x)/x."""
    x[abs(x) < 1e-8] = 1e-8
    j = 2 * sp.jn(1, np.pi*x)/(np.pi * x)
    j[abs(x) < 1e-8] = 1.0
    return j


def db(p):
    """Decibel from pressure."""
    return 20 * np.log10(abs(p))


# Calculations
def calculate_beamprofile(plot_region, aperture):
    """Calculate beam profile for specified aperture.

    Arguments
    ---------
    plot_region     PlotRegion  Region to calculate and plot results
    aperture        Aperture    Transducer aperture

    Output
    ------
    directivity     Directivity     Calcuated directivity patterns
    """
    width_lambda = aperture.width / aperture.wavelength()
    height_lambda = aperture.height / aperture.wavelength()

    theta = plot_region.theta()
    phi = plot_region.phi()
    theta_xy = plot_region.theta_xy()
    phi_xy = plot_region.phi_xy()
    theta_r = plot_region.theta_r()

    directivity = Directivity()
    if aperture.shape.casefold() == 'circular':
        directivity.theta = jinc(width_lambda * np.sin(theta))
        directivity.phi = directivity.theta
        directivity.xy = jinc(width_lambda * np.sin(theta_r))
    else:
        directivity.theta = np.sinc(width_lambda * np.sin(theta))  # Azimuth
        directivity.phi = np.sinc(height_lambda * np.sin(phi))     # Elevation
        directivity.xy = np.sinc(width_lambda * np.sin(theta_xy)) \
            * np.sinc(height_lambda * np.sin(phi_xy))  # 2D lateral intensity

    return directivity


# %%Plotting
def draw_element(aperture, ax):
    """Draw image of aperture shape.

    aperture    Aperture    Transduce aperture
    ax          Axes        Handle to axis for aperture plot
    """
    elementcolor = 'maroon'
    w_mm = aperture.width*1e3
    h_mm = aperture.height*1e3
    if aperture.shape.casefold() == 'circular':
        illustration = patches.Circle((0, 0), w_mm/2,
                                      fill=True,
                                      color=elementcolor,
                                      linewidth=2)
    else:
        illustration = patches.Rectangle((-w_mm/2, -h_mm/2), w_mm, h_mm,
                                         fill=True,
                                         color=elementcolor,
                                         linewidth=2)
    rectsize = 10
    ax.add_patch(illustration)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-rectsize, rectsize)
    ax.set_ylim(-rectsize, rectsize)
    ax.set_xlabel("Azimuth [mm]")
    ax.set_ylabel("Elevation [mm]")
    ax.set_title("Element shape")
    return 0


def plot_beamprofile(angle, directivity, title, ax, swap=False):
    """Plot beam profiles in graphs.

    Arguments
    ---------
    angle           Numpy 1D array  Angles to plot [Deg]
    directivity     Directivity     Calculated directivity patterns
    title           String          Title for plot
    ax              Axes            Axis to plot graph in
    swap            Boolean         Swap x- and y-axes
    """
    angle_lim = (-90, 90)
    db_lim = (-40, 0)
    angle_label = ("Angle [Deg]")
    theta_label = ("Power [dB rel. max]")

    if not swap:
        x = np.degrees(angle)
        y = db(directivity)
        x_lim = angle_lim
        y_lim = db_lim
        x_label = angle_label
        y_label = theta_label
    else:
        y = np.degrees(angle)
        x = db(directivity)
        y_lim = angle_lim
        x_lim = db_lim
        y_label = angle_label
        x_label = theta_label

    ax.plot(x, y)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.grid(True, axis='both')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # plt.title(title)

    return ax


def plot_lateral_intensity(directivity, plotregion, ax):
    """Plot intensity in lateral plane (xy) at fixed distance z.

    Arguments
    ---------
    directivity     Directivity     Calculated directivity patterns
    plot_region     PlotRegion      Region to plot
    ax              Axes            Axis to plot graph in
    """
    x_max = plotregion.x_max*1e3   # Convert to mm for plotting
    z = plotregion.z*1e3

    intensity_image = ax.imshow(db(directivity),
                                vmin=-40,
                                extent=[-x_max, x_max, -x_max, x_max],
                                cmap='viridis')

    plt.colorbar(intensity_image)
    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-x_max, x_max)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("Azimuth [mm]")
    ax.set_ylabel("Elevation [mm]")
    ax.set_title("Intensity at %0.0f mm [dB rel. max]" % z)
    return 0


def calculate_plot(width, height, frequency, shape):
    """Calculate and plot results.

    Main function. Dislays aperture shape, calculates and plots
    directivity patterns

    Attributes
    ----------
    width       Float       Element width (azimuth) or diameter [m]
    height      Float       Element height (elevation) [m]
    frequency   Float       Frequency [Hz]
    shape       Float       Element shape, "Rectangular", "Circular"
    """
    aperture = Aperture()

    aperture.shape = shape
    aperture.width = width*1e-3   # m  Element width
    aperture.height = height*1e-3   # m  Element width
    aperture.frequency = frequency*1e6    # Hz Frequency

    # Calculate directivity
    plotregion = PlotRegion(THETA_MAX, X_MAX, Z_REF)
    directivity = calculate_beamprofile(plotregion, aperture)

    # Plot results ---
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    draw_element(aperture, axs[0, 0])
    plot_lateral_intensity(directivity.xy,
                           plotregion,
                           axs[1, 1])
    plot_beamprofile(plotregion.theta(),
                     directivity.theta,
                     "Azimuth (x)",   axs[0, 1], swap=False)
    plot_beamprofile(plotregion.phi(),
                     directivity.phi,
                     "Elevation (y)", axs[1, 0], swap=True)

    return aperture, plotregion, directivity
