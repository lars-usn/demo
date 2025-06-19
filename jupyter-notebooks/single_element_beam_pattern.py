"""Run this to import libraries and define internal functions."""

# Libraries
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
import ipywidgets as widgets

# Mathematical functions


def jinc(x):
    """jinc-function, Bessel-version of sinc, 2J(x)/x."""
    x[abs(x) < 1e-8] = 1e-8
    j = 2 * sp.jn(1, np.pi*x)/(np.pi * x)
    j[abs(x) < 1e-8] = 1.0
    return j


def db(p, p_ref=1e-6):
    """Decibel from pressure."""
    if p_ref == 0:
        p_ref = np.max(p)

    return 20 * np.log10(abs(p/p_ref))


def find_width(x, y, y_lim):
    """Find width of function y(x) where y>y_lim.

    Assumes y(x) has a unique peak.
    Primitive version, no interpolation yet.
    """
    k = np.flatnonzero(y > y_lim)
    d_x = x[k[-1]] - x[k[0]]

    return d_x


# Main object
class SingleElement:
    """Define, calculate, and display transducer beam profile."""

    def __init__(self):
        self.circular = False    # Circular or rectangular element
        self.width = 100e-3      # m   Element width (azimuth, x) or diameter
        self.height = 100e-3     # m   Element height (elevation, y)

        self.frequency = 100e3  # Hz  Ultrasound frequency
        self.c = 1500         # m/s  Speed of sound

        self.theta_max = 90   # deg  Max. angle to calculate
        self.d_max = 200e-3   # m    Max. lateral on element display
        self.z_ref = 1        # m    Reference distance for lateral profile
        self.db_min = -30     # dB   Min. on dB-scales

        self.elementcolor = 'navy'
        self.element_background = 'lightgray'
        self.text_face = 'whitesmoke'

        self.ax, self.fig = self._initialise_graphs()

    def x_max(self):
        """Max. scale on intensity plot."""
        return 5*self.d_max

    def wavelength(self):
        """Calculate acoustic wavelenght."""
        return self.c/self.frequency

    def theta(self):
        """Azimuth (x) angle ."""
        theta_m = np.radians(self.theta_max)
        return np.linspace(-theta_m, theta_m, 501)

    def phi(self):
        """Elevation (y) angle."""
        phi_m = np.radians(self.theta_max)
        return np.linspace(-phi_m, phi_m, 501)

    def xy(self):
        """Lateral region to plot, plane at fixed axial distance."""
        pts = np.linspace(-self.x_max(), self.x_max(), 201)
        return np.meshgrid(pts, pts)

    def theta_xy(self):
        """Azimuth angles for xy-positions at distance z."""
        return np.arctan2(self.xy()[0], self.z_ref)

    def phi_xy(self):
        """Elevation angles for xy-positions at distance z."""
        return np.arctan2(self.xy()[1], self.z_ref)

    def theta_r(self):
        """Angle with z-axis for (xyz)-positions."""
        return np.arctan2(np.sqrt(self.xy()[0]**2+self.xy()[1]**2), self.z_ref)

    def w_lambda(self):
        """Aperture width relative to wavelength."""
        return self.width / self.wavelength()

    def h_lambda(self):
        """Aperture height relative to wavelength."""
        return self.height / self.wavelength()

    def display_result(self):
        """Display result in graphs."""
        for ax in self.ax.values():
            self._remove_artists(ax)

        # Element image
        self._draw_element(self.ax["element"])

        # Beam profiles
        if self.circular:
            p_x = self._p_circ(theta=self.theta())
            p_y = None
            p_xy = self._p_circ(theta=self.theta_r())
        else:
            p_x = self._p_rect(theta=self.theta(), phi=0)
            p_y = self._p_rect(theta=0, phi=self.phi())
            p_xy = self._p_rect(theta=self.theta_xy(), phi=self.phi_xy())

        theta_deg = np.degrees(self.theta())
        self.ax['azimuth'].plot(theta_deg, db(p_x, p_ref=0), color='C0')

        if not (p_y is None):
            phi_deg = np.degrees(self.phi())
            self.ax['elevation'].plot(phi_deg, db(p_y, p_ref=0), color='C0')

        # Lateral plane
        x = self.xy()[0][0, :]*1e3
        y = self.xy()[1][:, 0]*1e3
        im = self.ax['intensity'].pcolormesh(x, y, db(p_xy, p_ref=0),
                                             vmin=self.db_min,
                                             cmap='magma')
        self.cbar = self.fig.colorbar(im, ax=self.ax['intensity'])

        # -6 dB limits
        px_lim = 0.5*max(p_x)
        d_theta = find_width(self.theta(), p_x, px_lim)
        d_theta = np.degrees(d_theta)

        if not (self.circular):
            py_lim = 0.5*max(p_y)
            d_phi = find_width(self.phi(), p_y, py_lim)
            d_phi = np.degrees(d_phi)

        # Text box with results
        resulttext_1 = (fr'Frequency $f$: {self.frequency/1e3:.0f} kHz'
                        '\n'
                        fr'Wavelength $\lambda$: {self.wavelength()*1e3:.0f} mm'
                        '\n\n')

        if self.circular:
            resulttext_2 = ('Circular element\n'
                            fr'Diameter: {self.width*1e3:.1f} mm = {self.w_lambda():.1f} $\lambda$'
                            '\n'
                            fr'Opening angle (-6 dB): {d_theta:.1f}$^\circ$')
        else:
            resulttext_2 = (
                'Rectangular element\n'
                fr'Width (azimuth, x): {self.width*1e3:.1f} mm = {self.w_lambda():.1f} $\lambda$'
                '\n'
                fr'Heigth (azimuth, y): {self.height*1e3:.1f} mm = {self.h_lambda():.1f} $\lambda$'
                '\n'
                f'Opening angles (-6 dB)\n'
                fr'    Azimuth (x): {d_theta:.1f}$^\circ$'
                '\n'
                fr'    Elevation (y): {d_phi:.1f}$^\circ$')

        self._remove_fig_text(self.fig.texts)
        self.fig.text(0.2, 0.1, resulttext_1 + resulttext_2,
                      fontsize='medium',
                      bbox={'facecolor': self.text_face,
                            'boxstyle': 'Round',
                            'pad': 2})

        return

    ###################################################################
    # Non-public methods

    def _remove_artists(self, ax):
        """Clear axis of all old atrists."""
        for art in list(ax.lines):
            art.remove()
        for art in list(ax.collections):
            art.remove()
        for art in list(ax.patches):
            art.remove()
        for art in list(ax.texts):
            art.remove()

        try:
            self.cbar.remove()
        except:
            pass

        return 0

    def _remove_fig_text(self, fig):
        for art in list(self.fig.texts):
            art.remove()

        return 0

    def _draw_element(self, ax):
        """Draw image of aperture in specified axis (ax)."""
        w_mm = self.width*1e3
        h_mm = self.height*1e3
        if self.circular:
            illustration = patches.Circle((0, 0), w_mm/2,
                                          fill=True,
                                          color=self.elementcolor)
        else:
            illustration = patches.Rectangle((-w_mm/2, -h_mm/2), w_mm, h_mm,
                                             fill=True,
                                             color=self.elementcolor)
        ax.add_patch(illustration)

        return 0

    def _initialise_graphs(self):
        """Initialise graphs for signals and spectra."""
        plt.close('all')
        fig = plt.figure(figsize=[12, 6],
                         constrained_layout=True,
                         num='Single Element Beamprofile')

        ax = {'element':  fig.add_subplot(2, 3, 1),
              'intensity': fig.add_subplot(2, 3, 2),
              'azimuth':  fig.add_subplot(2, 3, 3),
              'elevation': fig.add_subplot(2, 3, 6),
              }

        # Lateral figures: Element drawing and intensity plot
        for a in [ax['element'], ax['intensity']]:
            a.set(box_aspect=1,
                  xlabel='Azimuth [mm]',
                  ylabel='Elevation [mm]')

        element_max = self.d_max * 1e3 * np.array([-1, 1])
        ax["element"].set(title='Element shape',
                          xlim=element_max,
                          ylim=element_max,
                          facecolor=self.element_background)

        intensity_max = self.x_max()*1e3*np.array([-1, 1])
        intensity_title = f'Intensity at {self.z_ref:.2f} m [dB re. max]'
        ax["intensity"].set(title=intensity_title,
                            xlim=intensity_max,
                            ylim=intensity_max,
                            facecolor=self.element_background)

        # Directivity graphs
        angle_lim = self.theta_max * np.array([-1, 1])
        db_lim = (self.db_min, 0)

        for a in [ax['azimuth'], ax['elevation']]:
            a.set(xlim=angle_lim,
                  ylim=db_lim,
                  xlabel='Angle [Deg]',
                  ylabel='Power [dB re. max]',
                  )

            a.xaxis.set_major_locator(MultipleLocator(30))
            a.xaxis.set_minor_locator(MultipleLocator(10))
            a.yaxis.set_major_locator(MultipleLocator(3))
            a.yaxis.set_minor_locator(MultipleLocator(1))

            a.grid(visible=True, which='both', axis='x')
            a.grid(visible=True, which='major', axis='y')

        ax['azimuth'].set_title('Azimuth beam profile')
        ax['elevation'].set_title('Elevation beam profile')

        return ax, fig

    def _p_circ(self, theta):
        """Calculate pressure field from circular element, 1D or 2D."""
        p = jinc(self.w_lambda() * np.sin(theta))
        return p

    def _p_rect(self, theta=0, phi=0):
        """Calculate pressure field from rectangular element, 1D or 2D."""
        p = np.sinc(self.w_lambda() * np.sin(theta)) \
            * np.sinc(self.h_lambda() * np.sin(phi))

        return p
