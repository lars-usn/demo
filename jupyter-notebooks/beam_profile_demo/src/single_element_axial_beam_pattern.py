"""Run this to import libraries and define internal functions."""

# Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import ipywidgets as widgets

import curve_analysis as ca


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
class Transducer():
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

        self.y_lim = 0.5      # Ralative limit for markers
        self.lim_text = '-6 dB'

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
        """Lateral points(azimuth, x)."""
        return self._points()[1]

    def z(self):
        """Axial points(z)."""
        return self._points()[0]

    def r(self):
        """Distance from aperture centre to point(z, x)."""
        return np.sqrt(self.z()**2 + self.x()**2)

    def theta(self):
        """Azimuth(x) angle to point(z, x)."""
        return np.atan2(self.x(), self.z())

    def p(self):
        """Calculate pressure field."""
        if self.circular:
            p = 1/self.r() * ca.jinc(self.w_lambda() * np.sin(self.theta()))
        else:
            p = 1/self.r() * np.sinc(self.w_lambda() * np.sin(self.theta()))
        return p

    def w_lambda(self):
        """Aperture width relative to wavelength."""
        return self.width / self.wavelength()

    def display_result(self):
        """Display result in graphs."""
        for ax in self.ax.values():
            ca.remove_artists(ax)

        try:
            self.cbar.remove()
        except Exception:
            pass

        marker_color = 'white'
        element_color = 'navy'

        # Intensity image
        x = self.x()[:, 0]
        z = self.z()[0, :]
        im = self.ax['intensity'].pcolormesh(z, x, ca.db(self.p(), p_ref=0),
                                             vmin=self.db_min,
                                             cmap='magma')

        self.ax['intensity'].axvline(x=self.z_ref,
                                     color=marker_color,
                                     linestyle='dotted')

        y_element = self.width/2
        self.ax['intensity'].axhspan(-y_element, y_element, xmax=0.01,
                                     color=element_color)

        for w in [-self.width, self.width]:
            self.ax['intensity'].plot([0.0, self.z_r()],
                                      w/2*np.array([1, 1]),
                                      color=element_color,
                                      linestyle='dotted')

        self.cbar = self.fig.colorbar(im, ax=self.ax['intensity'])
        ca.db_colorbar(self.cbar, db_sep=6)

        x_line = np.array([0, self.z_max])
        y_line = np.array([0, self.z_max*np.tan(self.d_theta()/2)])
        self.ax['intensity'].plot(x_line, y_line, x_line, -y_line,
                                  color=marker_color,
                                  linestyle='dotted')

        # Lateral beam profile
        k_ref = np.argmin(abs(z-self.z_ref))
        p = self.p()[:, k_ref]
        p_db = ca.db(p, p_ref=self.p().max())
        self.ax['azimuth'].plot(x, p_db, color='C0')

        indicator_line = {'color': 'C1',
                          'linestyle': 'solid'}

        self.ax['azimuth'].axhline(y=p_db.max()+ca.db(self.y_lim, p_ref=1),
                                   **indicator_line)

        # Find reference values
        ref = ca.Refpoints(x=x, y=p)
        xl, _ = ref.ref_values(y_rel=self.y_lim)   # Beam width limits
        self.dx = xl[1] - xl[0]

        self.x_sidelobe, self.y_sidelobe = ref.sidelobe()   # Highest sidelobe
        self.db_sidelobe = ca.db(self.y_sidelobe, p_ref=p.max())

        for x in xl:
            self.ax['azimuth'].axvline(x=x, **indicator_line)

        # Text box with results
        self._show_resulttext()

        return

    def display_interactively(self, circular=True, frequency=100,
                              width=100, z_ref=1.0):
        """Scale inputs and display results.

        For interactive operation with dimensions in mm and frequency in kHz.
        Existing values are used if a parameter is omitted.

        Parameters
        ----------
        circular: boolean
            Circular aperture or 1D line transducer
        frequency: float
            Frequency in kHz
        widh: float
            Element width(azimuth, x) in mm
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
        """Mesh of points(z, x)."""
        return np.meshgrid(np.linspace(self.z_r(), self.z_max, 500),
                           np.linspace(-self.x_max, self.x_max, 301))

    def _initialise_graphs(self):
        """Initialise result graphs."""
        plt.close('all')
        fig = plt.figure(figsize=[14, 6],
                         constrained_layout=True,
                         num='Single Element Beamprofile')

        ca.add_logo(fig)

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

        ca.db_axis(ax['azimuth'], db_min=self.db_min, db_max=0, db_sep=6)

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
                      'Opening angle: '
                      fr'{np.degrees(self.d_theta()):.1f}$^\circ$'
                      '\n\n'
                      fr'Reference distance {self.z_ref:.2} m'
                      '\n'
                      fr'    Beam width ({self.lim_text}): '
                      fr'{1e3*self.dx:.0f} mm'
                      )

        if np.isnan(self.x_sidelobe):
            sidelobetext = ''
        else:
            sidelobetext = ('    Highest sidelobe: '
                            fr'$x$ = {self.x_sidelobe:.2f} m, '
                            fr'{self.db_sidelobe:.1f} dB'
                            )

        ca.set_fig_text(self.fig,
                        headertext + '\n' + resulttext + '\n' + sidelobetext,
                        xpos=0.07, ypos=0.17)

        return

    def _create_widgets(self):
        """Create widgets for interactive operation."""
        widget_style = {'description_width': 'initial'}

        title = 'Axial Beam-profile from Single Element Transducer'
        title_widget = widgets.Label(title, style=dict(font_weight='bold'))

        shape_widget = widgets.Dropdown(options=[
            ('Line transducer (1D)', False),
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
