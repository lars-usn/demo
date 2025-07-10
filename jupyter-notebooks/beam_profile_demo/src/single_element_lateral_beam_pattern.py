"""Run this to import libraries and define internal functions."""

# Libraries
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.ticker import MultipleLocator
from matplotlib import transforms

import ipywidgets as widgets

import curve_analysis as ca


# Mathematical functions
def find_width(x, y, y_lim):
    """Find width of function y(x) where y>y_lim.

    Assumes y(x) has a unique peak.
    Primitive version, no interpolation yet.
    """
    k = np.flatnonzero(y > y_lim)
    d_x = x[k[-1]] - x[k[0]]

    return d_x


# Calculate and display lateral bem profile
class Transducer():
    """Define, calculate, and display transducer beam profile."""

    def __init__(self, create_graphs=False, create_widgets=False):
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

        if create_graphs:
            self.ax, self.fig = self._initialise_graphs()
        if create_widgets:
            self.widget_layout, self.widget = self._create_widgets()

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

        db_marker = -6
        marker_color = 'red'

        # Beam profiles
        if self.circular:
            p_x = self._p_circ(theta=self.theta())
            p_y = None
            p_xy = self._p_circ(theta=self.theta_r())
        else:
            p_x = self._p_rect(theta=self.theta(), phi=0)
            p_y = self._p_rect(theta=0, phi=self.phi())
            p_xy = self._p_rect(theta=self.theta_xy(), phi=self.phi_xy())

        self._plot_beamprofile(self.ax['azimuth'],
                               np.degrees(self.theta()),
                               ca.db(p_x, p_ref=0),
                               y_marker=db_marker)

        if not (p_y is None):
            self._plot_beamprofile(self.ax['elevation'],
                                   np.degrees(self.phi()),
                                   ca.db(p_y, p_ref=0),
                                   y_marker=db_marker)

        # Lateral plane
        x_mm = self.xy()[0][0, :]*1e3
        y_mm = self.xy()[1][:, 0]*1e3
        im = self.ax['intensity'].pcolormesh(x_mm, y_mm,
                                             ca.db(p_xy, p_ref=0),
                                             vmin=self.db_min,
                                             cmap='magma')

        self.ax['intensity'].contour(x_mm, y_mm,
                                     ca.db(p_xy, p_ref=0),
                                     levels=[db_marker],
                                     colors=marker_color,
                                     linestyles='solid',
                                     alpha=0.9)

        self.cbar = self.fig.colorbar(im, ax=self.ax['intensity'])
        ca.db_colorbar(self.cbar, db_sep=6)

        # -6 dB limits
        ref = ca.Refpoints(x=self.theta(), y=p_x)
        self.d_theta, _ = ref.lobe_width(y_rel=0.5)

        if not (self.circular):
            ref = ca.Refpoints(x=self.theta(), y=p_y)
            self.d_phi, _ = ref.lobe_width(y_rel=0.5)

        # Text box with results
        self._show_resulttext()

        return

    def display_interactively(self, circular=None, frequency=None,
                              width=None, height=None):
        """Scale inputs and  display results.

        For interactive operation with  dimensions in mm and frequency in kHz.
        Existing values are used if a parameter is omitted.

        Parameters
        ----------
        circular: boolean, optional
            Circular (True) or rectangular (False) aperture
        frequency: float, optional
            Frequency in kHz
        widh: float, optional
            Element width (azimuth, x) in mm
        height: float, optional
            Element height (elevation, y) in mm
        """
        if circular is not None:
            self.circular = circular
        if frequency is not None:
            self.frequency = 1e3*frequency
        if width is not None:
            self.width = 1e-3*width
        if height is not None:
            self.height = 1e-3*height

        self.display_result()

        return

    ###################################################################
    # Non-public methods

    # Calculations, internal
    def _p_circ(self, theta):
        """Calculate pressure field from circular element, 1D or 2D."""
        p = ca.jinc(self.w_lambda() * np.sin(theta))
        return p

    def _p_rect(self, theta=0, phi=0):
        """Calculate pressure field from rectangular element, 1D or 2D."""
        p = np.sinc(self.w_lambda() * np.sin(theta)) \
            * np.sinc(self.h_lambda() * np.sin(phi))

        return p

    # Layout
    def _add_logo(self, fig):
        """Add logo image to result figure."""
        try:
            img = mpimg.imread('usn-logo-purple.png')
            ax_logo = fig.add_axes([0.02, 0.02, 0.2, 0.2], anchor='SW')
            ax_logo.imshow(img)
            ax_logo.axis('off')
        except Exception:
            pass

        return

    # Graphs and results
    def _remove_artists(self, ax):
        """Clear axis of all old artists."""
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
        except Exception:
            pass

        return

    def _remove_fig_text(self, fig):
        """Remove all existing text from figure."""
        for art in list(self.fig.texts):
            art.remove()

        return 0

    def _plot_beamprofile(self, ax, angle, y, y_marker=None,
                          line_color='C0', marker_color="C1"):
        """Plot beam profile graph."""
        ax.plot(angle, y, color=line_color)
        if not (y_marker is None):
            ax.axhline(y=y_marker, color=marker_color)

        return 0

    def _draw_element(self, ax):
        """Draw image of aperture in specified axis."""
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

    def _show_resulttext(self):
        """Create text box and fill with results."""
        resulttext_1 = (fr'Frequency  $f$= '
                        fr'{self.frequency/1e3:.0f} kHz'
                        '\n'
                        r'Wavelength  $\lambda$='
                        f'{self.wavelength()*1e3:.0f} mm'
                        '\n\n')

        if self.circular:
            resulttext_2 = ('Circular element\n'
                            f'  Diameter  D= '
                            f'{self.width*1e3:.1f} mm = '
                            fr'{self.w_lambda():.1f} $\lambda$'
                            '\n'
                            fr'  Opening angle (-6 dB)  $\theta_0$= '
                            fr'{np.degrees(self.d_theta):.1f}$^\circ$')
        else:
            resulttext_2 = ('Rectangular element\n'
                            '  Width (azimuth, x) $w$= '
                            fr'{self.width*1e3:.1f} mm = '
                            fr'{self.w_lambda():.1f} $\lambda$'
                            '\n'
                            '  Heigth (elevation, y) $h$= '
                            fr'{self.height*1e3:.1f} mm = '
                            fr'{self.h_lambda():.1f} $\lambda$'
                            '\n'
                            f'Opening angles (-6 dB)\n'
                            fr'  Azimuth $\theta_0$= '
                            fr'{np.degrees(self.d_theta):.1f}$^\circ$'
                            '\n'
                            fr'  Elevation $\phi_0$= '
                            fr'{np.degrees(self.d_phi):.1f}$^\circ$')

        self._remove_fig_text(self.fig.texts)
        self.fig.text(0.20, 0.15, resulttext_1 + resulttext_2,
                      fontsize='medium',
                      bbox={'facecolor': self.text_face,
                            'boxstyle': 'Round',
                            'pad': 1})

        return

    def _initialise_graphs(self):
        """Initialise result graphs."""
        plt.close('all')
        fig = plt.figure(figsize=[12, 6],
                         constrained_layout=True,
                         num='Single Element Beamprofile')

        self._add_logo(fig)

        ax = {'element': fig.add_subplot(2, 3, 1),
              'intensity': fig.add_subplot(2, 3, 2),
              'azimuth': fig.add_subplot(2, 3, 3),
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
        for a in [ax['azimuth'], ax['elevation']]:
            a.set(box_aspect=1,
                  xlim=angle_lim,
                  xlabel='Angle [Deg]',
                  ylabel='Power [dB re. max]')

            ca.db_axis(a, db_min=self.db_min, db_max=0, db_sep=6)

            a.xaxis.set_major_locator(MultipleLocator(30))
            a.xaxis.set_minor_locator(MultipleLocator(10))
            a.grid(visible=True, which='both', axis='x')

        ax['azimuth'].set_title('Azimuth beam profile')
        ax['elevation'].set_title('Elevation beam profile')

        return ax, fig

    # Interactive widgets
    def _create_widgets(self):
        """Create widgets for interactive operation."""
        widget_style = {'description_width': 'initial'}

        title = 'Lateral Beam-profile from Single Element Transducer'
        title_widget = widgets.Label(title, style=dict(font_weight='bold'))

        shape_widget = widgets.Dropdown(options=[('Rectangular', False),
                                                 ('Circular', True)],
                                        value=False,
                                        description='Shape',
                                        layout=widgets.Layout(width='80%'))

        label = ['Frequency', 'Width (Diameter)', 'Height']
        label_widget = [widgets.Label(labeltext,
                                      layout=widgets.Layout(width='20%'))
                        for labeltext in label]

        layout_settings = {'continuous_update': True,
                           'layout': widgets.Layout(width='80%'),
                           'style': widget_style}

        frequency_widget = widgets.FloatSlider(min=1, max=400,
                                               value=100, step=1,
                                               readout_format='.0f',
                                               description='[kHz]',
                                               **layout_settings)

        size_widget = [widgets.FloatSlider(min=1, max=400,
                                           value=100, step=1,
                                           readout_format='.0f',
                                           description='[mm]',
                                           **layout_settings)
                       for k in range(2)]

        parameter_widget = [frequency_widget] + size_widget
        parameter_line = [widgets.HBox([label_widget[k], parameter_widget[k]])
                          for k in range(len(label))]

        col = [widgets.VBox([shape_widget],
                            layout=widgets.Layout(width='30%')),
               widgets.VBox(parameter_line,
                            layout=widgets.Layout(width='70%'))]

        grid = widgets.HBox(col, layout=widgets.Layout(width='90%'))
        widget_layout = widgets.VBox([title_widget, grid],
                                     layout=widgets.Layout(width='90%'))

        widget = {'circular': shape_widget,
                  'frequency': frequency_widget,
                  'width': size_widget[0],
                  'height': size_widget[1]
                  }

        return widget_layout, widget
