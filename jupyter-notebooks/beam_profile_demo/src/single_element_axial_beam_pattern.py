"""Run this to import libraries and define internal functions."""

# Libraries
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec
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


def db_axis(ax, db_min=-40, db_max=0, db_sep=6):
    """Configure dB-scaled axis.

    Parameters
    ----------
    ax: axis object
        Axis to configure
    db_min: float
        Minimum on dB-axis
    db_max: float
        Maximum on dB-axis
    db_sep: float
        Separation between major ticks
    """
    ax.set_ylim(db_min, db_max)

    ax.yaxis.set_major_locator(MultipleLocator(db_sep))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(visible=True, which='major', axis='y')

    return


def db_colorbar(cbar, db_sep=6):
    """Configure dB-scaled colorbar.

    Parameters
    ----------
    cbar: Colorbar object
        Colorbar to configure
    db_sep: float
        Separation between major ticks
    """
    tick_min = (cbar.vmin // db_sep) * db_sep
    db_ticks = np.arange(tick_min, cbar.vmax+db_sep, db_sep)
    cbar.set_ticks(db_ticks)
    cbar.minorticks_off()

    return


def find_width(x, y, y_lim):
    """Find width of function y(x) where y>y_lim.

    Assumes y(x) has a unique peak.
    Linear interpolation to impo
    """
    k = np.flatnonzero(y > y_lim)
    kh = [k[-1], k[-1]+1]
    kl = [k[0], k[0]-1]

    xl = [x[k[0]] + (y_lim - y[k[0]])
          * (x[k[1]] - x[k[0]]) / (y[k[1]] - y[k[0]])
          for k in [kl, kh]]

    return xl


# Calculate and display lateral bem profile
class SingleElement():
    """Define, calculate, and display transducer beam profile."""

    def __init__(self, create_graphs=False, create_widgets=False):
        self.circular = True
        self.width = 100e-3     # m   Element width (azimuth, x) or diameter
        self.frequency = 100e3  # Hz  Ultrasound frequency
        self.c = 1500           # m/s  Speed of sound

        self.z_max = 10.0       # m    Max. axial distance calculate
        self.x_max = 1.0        # m    Max. lateral distance calculate

        self.z_ref = 4.0      # m    Reference distance for lateral profile
        self.db_min = -60     # dB   Min. on dB-scales

        self.element_background = 'lightgray'
        self.text_face = 'whitesmoke'

        if create_graphs:
            self.ax, self.fig = self._initialise_graphs()
        if create_widgets:
            self.widget_layout, self.widget = self._create_widgets()

    def wavelength(self):
        """Calculate acoustic wavelenght."""
        return self.c/self.frequency

    def d_theta(self):
        """Calculate opening angle from theory, two-sided, -6 dB."""
        if self.circular:
            x_6 = 0.705   # 6 dB limit, circular aperture
        else:
            x_6 = 0.603   # 6 dB limit, line (rectangular) aperture

        return 2*np.asin(x_6 * self.wavelength()/self.width)

    def z_r(self):
        """Rayleigh distance, far-field limit."""
        return self.width**2/(2*self.wavelength())

    def x(self):
        """Lateral points (azimuth, x)."""
        return self._points()[1]

    def z(self):
        """Axial points (z)."""
        return self._points()[0]

    def r(self):
        """Distance from aperture centre to point (z,x)."""
        return np.sqrt(self.z()**2 + self.x()**2)

    def theta(self):
        """Azimuth (x) angle to point (z,x)."""
        return np.atan2(self.x(), self.z())

    def p(self):
        """Calculate pressure field."""
        if self.circular:
            p = 1/self.r() * jinc(self.w_lambda() * np.sin(self.theta()))
        else:
            p = 1/self.r() * np.sinc(self.w_lambda() * np.sin(self.theta()))
        return p

    def w_lambda(self):
        """Aperture width relative to wavelength."""
        return self.width / self.wavelength()

    def display_result(self):
        """Display result in graphs."""
        for ax in self.ax.values():
            self._remove_artists(ax)

        marker_color = 'white'
        element_color = 'maroon'

        # Intensity image
        x = self.x()[:, 0]
        z = self.z()[0, :]
        im = self.ax['intensity'].pcolormesh(z, x, db(self.p(), p_ref=0),
                                             vmin=self.db_min,
                                             cmap='magma')

        self.ax['intensity'].axvline(x=self.z_ref,
                                     color=marker_color,
                                     linestyle='dotted')

        self.ax['intensity'].plot([0.0, 0.0], self.width/2*np.array([-1, 1]),
                                  color=element_color,
                                  linewidth=4)

        for w in [-self.width, self.width]:
            self.ax['intensity'].plot([0.0, self.z_r()],
                                      w/2*np.array([1, 1]),
                                      color=element_color,
                                      linestyle='dotted')

        self.cbar = self.fig.colorbar(im, ax=self.ax['intensity'])
        db_colorbar(self.cbar, db_sep=6)

        x_line = np.array([0, self.z_max])
        y_line = np.array([0, self.z_max*np.tan(self.d_theta()/2)])
        self.ax['intensity'].plot(x_line, y_line, x_line, -y_line,
                                  color=marker_color,
                                  linestyle='dotted')

        # Lateral beam profile
        k_ref = np.argmin(abs(z-self.z_ref))
        p_db = db(self.p()[:, k_ref], p_ref=self.p().max())
        self.ax['azimuth'].plot(x, p_db, color='C0')

        indicator_line = {'color': 'red',
                          'linestyle': 'dotted'}

        self.ax['azimuth'].axhline(y=p_db.max()-6, **indicator_line)

        self.xl = find_width(x, p_db, p_db.max()-6)
        self.dx = self.xl[1] - self.xl[0]

        for x in self.xl:
            self.ax['azimuth'].axvline(x=x, **indicator_line)

        # Text box with results
        self._show_resulttext()

        return

    def display_interactively(self, circular=True, frequency=100,
                              width=100, z_ref=1.0):
        """Scale inputs and  display results.

        For interactive operation with  dimensions in mm and frequency in kHz.
        Existing values are used if a parameter is omitted.

        Parameters
        ----------
        circular: boolean
            Circular aperture or 1D line transducer
        frequency: float
            Frequency in kHz
        widh: float
            Element width (azimuth, x) in mm
        z_ref: float
            Reference distance for beam profile graph in m
        """
        self.circular = circular
        self.frequency = 1e3*frequency
        self.width = 1e-3*width
        self.z_ref = z_ref

        self.display_result()

        return

    ###################################################################
    # Non-public methods

    def _points(self):
        """Mesh of points (z,x)."""
        return np.meshgrid(np.linspace(self.z_r(), self.z_max, 500),
                           np.linspace(-self.x_max, self.x_max, 301))

    def _add_logo(self, fig):
        """Add logo image to result figure."""
        try:
            img = mpimg.imread('usn-logo-purple.png')
            ax_logo = fig.add_axes([0.05, 0.05, 0.2, 0.2], anchor='SW')
            ax_logo.imshow(img)
            ax_logo.axis('off')
        except Exception:
            pass

        return

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

    def _initialise_graphs(self):
        """Initialise result graphs."""
        plt.close('all')
        fig = plt.figure(figsize=[14, 6],
                         constrained_layout=True,
                         num='Single Element Beamprofile')

        self._add_logo(fig)

        gs = GridSpec(2, 2, figure=fig)
        ax = {'intensity': fig.add_subplot(gs[0, :]),
              'azimuth': fig.add_subplot(gs[1, 1])}

        # Lateral figures: Element drawing and intensity plot
        ax['intensity'].set(aspect='equal',
                            xlabel='Distance (z) [m]',
                            ylabel='Azimuth (x) [m]',
                            facecolor='lightgray')

        ax["intensity"].set(ylim=self.x_max*np.array([-1, 1]),
                            xlim=[0, self.z_max])

        # Directivity graphs
        ax['azimuth'].set(xlabel='Azimuth (x) [m]',
                          ylabel='Power [dB re. max]',
                          xlim=self.x_max*np.array([-1, 1]))

        db_axis(ax['azimuth'], db_min=self.db_min, db_max=0, db_sep=6)

        ax['azimuth'].grid(visible=True, which='both', axis='x')

        return ax, fig

    def _show_resulttext(self):
        """Create text box and fill with results."""
        if self.circular:
            headertext = 'Cicular aperture'
        else:
            headertext = 'Line transducer (1D)'

        resulttext = (fr'Frequency $f$ = {self.frequency/1e3:.0f} kHz'
                      '\n'
                      r'Wavelength $\lambda$ ='
                      f'{self.wavelength()*1e3:.0f} mm'
                      '\n\n'
                      f'Element width $w$ = {self.width*1e3:.0f} mm = '
                      fr'{self.w_lambda():.1f} $\lambda$'
                      '\n'
                      fr'Rayleigh distance $z_r$ = '
                      fr'{self.z_r():.1f} m'
                      '\n'
                      fr'Beam width (-6 dB) at distance {self.z_ref:.2} m: '
                      fr'{1e3*self.dx:.0f} mm'
                      '\n'
                      'Opening angle : '
                      fr'{np.degrees(self.d_theta()):.1f}$^\circ$')

        self._remove_fig_text(self.fig.texts)
        self.fig.text(0.07, 0.25, headertext + '\n' + resulttext,
                      fontsize='medium',
                      bbox={'facecolor': self.text_face,
                            'boxstyle': 'Round',
                            'pad': 1})

        return

    def _create_widgets(self):
        """Create widgets for interactive operation."""
        widget_style = {'description_width': 'initial'}

        title = 'Axial Beam-profile from Single Element Transducer'
        title_widget = widgets.Label(title, style=dict(font_weight='bold'))

        shape_widget = widgets.Dropdown(options=[('Line transducer (1D)', False),
                                                 ('Circular aperture', True)],
                                        value=True,
                                        description='Shape',
                                        layout=widgets.Layout(width='80%'))

        label = ['Frequency', 'Width', 'Ref. distance']
        label_widget = [widgets.Label(labeltext,
                                      layout=widgets.Layout(width='12%'))
                        for labeltext in label]

        layout_settings = {'continuous_update': True,
                           'layout': widgets.Layout(width='80%'),
                           'style': widget_style}

        frequency_widget = widgets.FloatSlider(min=1, max=400,
                                               value=100.0, step=1,
                                               readout_format='.0f',
                                               description='[kHz]',
                                               **layout_settings)

        size_widget = widgets.FloatSlider(min=1, max=400,
                                          value=100.0, step=1,
                                          readout_format='.0f',
                                          description='[mm]',
                                          **layout_settings)

        distance_widget = widgets.FloatSlider(min=1, max=10,
                                              value=2.0, step=0.1,
                                              readout_format='.1f',
                                              description='[m]',
                                              **layout_settings)

        parameter_widget = [frequency_widget, size_widget, distance_widget]
        parameter_line = [widgets.HBox([label_widget[k], parameter_widget[k]])
                          for k in range(len(label))]

        col = [widgets.VBox([shape_widget],
                            layout=widgets.Layout(width='25%')),
               widgets.VBox(parameter_line,
                            layout=widgets.Layout(width='70%'))]

        grid = widgets.HBox(col, layout=widgets.Layout(width='90%'))

        widget_layout = widgets.VBox([title_widget, grid],
                                     layout=widgets.Layout(width='90%'))

        widget = {'circular': shape_widget,
                  'frequency': frequency_widget,
                  'width': size_widget,
                  'distance': distance_widget
                  }

        return widget_layout, widget
