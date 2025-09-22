"""Calculate the beam pattern from a line array.

The array consists of N elements described either as points or rectangular
elements.
"""

# Python libraries
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
import ipywidgets as widgets

# Internal libraries
import curve_analysis as ca


# Class definitions
class WidgetLayout():
    """Container for widgets and layout."""

    def __init__(self, layout, widget):
        self.layout = layout
        self.widget = widget


class Array():
    """Define, calculate, and display transducer beam profile."""

    def __init__(self, create_widgets=False):

        # Transducer array definition
        self.n_elements = 64  # Number of elements in array
        self.kerf = 100e-6      # m   Kerf between elements
        self.pitch = 7.5e-3     # m   Pitch between elements
        self.frequency = 100e3  # Hz  Ultrasound frequency
        self.angle_s = 20       # deg Steering angle
        self.c = 1500           # m/s Speed of soundin load medium

        # Calculation settings
        self.z_ref = 20.0        # m    Reference depth
        self.y_lim = 0.5         # Relative limit for beamwidth

        # Display settings
        self.theta_max = 90    # deg  Max. angle to calculate
        self.x_max = 50.0      # m    Max. lateral dimension to calculate
        self.z_max = 100.0     # m    Max. depth to calculate
        self.db_range = 60     # dB   Dynamic gange on dB-scales
        self.db_gain = 6       # dB   Max. on dB-scales

        # Colors and markers
        self.text_face = 'whitesmoke'
        self.element_color = 'crimson'

        self.colormap = 'inferno'
        self.intensity_background = 'black'

        # Initialisation
        ax, fig = self._initialise_graphs()
        self.axes = ax
        self.fig = fig
        self.scale_axes()

        if create_widgets:
            self.widget = self._create_widgets()

    # === Calculated parameters ===========================
    def theta_s(self):
        """Steerning angle in radians.."""
        return np.radians(self.angle_s)

    def delay(self):
        """Delay between elements (no focusing)."""
        return -np.sin(self.theta_s()) * self.pitch / self.c

    def delay_array(self):
        """Delay over all elements."""
        n = np.arange(0, self.n_elements)
        tau = -n * self.delay()
        tau = tau - tau.min()
        return [tau, n+1]

    def width(self):
        """Width of element."""
        return self.pitch - self.kerf

    def wavelength(self):
        """Calculate acoustic wavelenght."""
        return self.c/self.frequency

    def d_aperture(self):
        """Width of aperture."""
        return self.n_elements * self.pitch - self.kerf

    def p_lambda(self):
        """Pitch relative to wavelength."""
        return self.pitch / self.wavelength()

    def w_lambda(self):
        """Element width relative to wavelength."""
        return self.width() / self.wavelength()

    def d_lambda(self):
        """Aperture width relative to wavelength."""
        return self.d_aperture() / self.wavelength()

    def z_r(self):
        """Rayleigh distance, far-field limit."""
        return self.d_aperture()**2/(2*self.wavelength())

    def z_c(self):
        """Limit reference distance to outside far-field limit."""
        return np.max([self.z_ref, self.z_r()])

    # === Axial plane ===================
    def x(self):
        """Lateral dimension for axial plot (x or y)."""
        return np.linspace(-self.x_max, self.x_max, 401)

    def z(self):
        """Axial dimension (depth) for axial plot (z)."""
        return np.linspace(self.z_r(), self.z_max, 600)

    def zx(self):
        """Axial plane (zx or zy) to plot."""
        pts = np.meshgrid(self.z(), self.x())
        return pts

    def r(self):
        """Distance from aperture centre for axial plot."""
        return np.sqrt(self.zx()[0]**2 + self.zx()[1]**2)

    def theta(self):
        """Azimuth(x) angle to point(z, x)."""
        return np.arctan2(self.zx()[1], self.zx()[0])

    def directivity_element(self, theta):
        """Directivity of one element."""
        return np.sinc(self.w_lambda() * np.sin(theta))

    def directivity_array_points(self, theta):
        """Directivity of arrray of points."""
        s_theta = np.sin(theta) - np.sin(self.theta_s())
        x = pi * self.p_lambda() * s_theta
        x[x == 0] = 1e-4   # Avoid 0/0 erroers
        d = np.sin(self.n_elements * x) / (self.n_elements * np.sin(x))

        return d

    def p_axial(self):
        """Calculate axial pressure field in the azimuth plane (zx)."""
        p0 = self.n_elements * self.width() / self.r()
        p = p0 * self.directivity_element(np.sin(self.theta())) * \
            self.directivity_array_points(np.sin(self.theta()))

        return p

    # === Commands =============================
    def display(self):
        """Display beam pattern in graphs."""
        self._remove_old_artists()
        ax = self.axes

        # Axial intensity
        p_ref = self.n_elements * self.width() / self.z_r()
        p_db = ca.db(self.p_axial(), p_ref=p_ref)

        im = ax['axial'].pcolormesh(self.x(), self.z(),  p_db.transpose(),
                                    clim=self._db_scale(),
                                    cmap=self.colormap)

        self.cbar = self.fig.colorbar(im, ax=ax['axial'])
        ca.db_colorbar(self.cbar, db_sep=6)

        # Element with lines extending to Rayleigh distance
        x_aperture = self.d_aperture() / 2
        ax['axial'].axhspan(-x_aperture, x_aperture, xmax=0.005,
                            color=self.element_color)

        for w in [-x_aperture, x_aperture]:
            ax['axial'].plot([0.0, self.z_r()],
                             w/2*np.array([1, 1]),
                             color=self.element_color,
                             linestyle='dotted')

        # Delay profile over array
        tau, n = self.delay_array()
        ax['delay'].stem(n, tau*1e6, 'C0')
        self.scale_axes()
        self._resulttext()

        # Beam profile, directivury function
        theta = np.linspace(-90, 90, 501)
        p_element = self.directivity_element(np.radians(theta))
        p_points = self.directivity_array_points(np.radians(theta))
        p_array = p_element * p_points

        ax['beamprofile'].plot(theta, ca.db(p_element, p_ref=1),
                               color='C0', linestyle='dashed')
        ax['beamprofile'].plot(theta, ca.db(p_array, p_ref=1),
                               color='C0', linestyle='solid')

        ax['beamprofile'].axvline(x=0, color='gray')
        return

    def scale_axes(self):
        """Change scales of all graphs."""
        ax = self.axes

        ax["axial"].set(xlim=self.x_max*np.array([-1, 1]),
                        ylim=(self.z_max, 0))

        delay_max = self.n_elements * self.pitch / self.c
        ax['delay'].set(xlim=(1, self.n_elements),
                        ylim=(0, 1e6 * delay_max))

        ax['beamprofile'].set(xlim=(90 * np.array([-1, 1])))
        ax['beamprofile'].xaxis.set_minor_locator(MultipleLocator(10))
        ax['beamprofile'].xaxis.set_major_locator(MultipleLocator(30))

        ca.db_axis(ax['beamprofile'], db_scale=(-42, 0), db_sep=6)

        return 0

    def interact(self,
                 n_elements=None,
                 frequency=None,
                 pitch=None,
                 angle_s=None,
                 db_range=None,
                 db_gain=None
                 ):
        """Scale inputs and  display results.

        For interactive operation with dimensions in mm and frequency in kHz.
        Existing values are used if a parameter is omitted.

        Parameters
        ----------
        n_elements: int, optional
            No. of elements in array
        frequency: float, optional
            Frequency in kHz
        pitch: float, optional
            Element pitch in mm
        angle_s: float, optional
            Steering angle in degrees
        db_range: float
            Range on dB-axes
        db_gain: float
            Maximum on dB-axes
        """
        if n_elements is not None:
            self.n_elements = n_elements
        if frequency is not None:
            self.frequency = 1e3*frequency
        if pitch is not None:
            self.pitch = 1e-3*pitch
        if angle_s is not None:
            self.angle_s = angle_s
        if db_range is not None:
            self.db_range = db_range
        if db_gain is not None:
            self.db_gain = db_gain

        # Display result in graphs
        self.display()

        return

    # === Non-public methods ==========================================

    # Graphs and results
    def _db_scale(self):
        db_lim = np.array([-self.db_range, 0]) - self.db_gain
        return db_lim

    def _resulttext(self):
        """Text box for lateral profile results."""
        header = (f'Frequency  $f$ = {self.frequency/1e3:.0f} kHz\n'
                  fr'Wavelength  $\lambda$ = {self.wavelength()*1e3:.1f} mm')

        array_text = (r'No. of elements $N_{el}$ = '
                      f'{self.n_elements:d} '
                      '\n'
                      f'Element width $w$ = {self.width()*1e3:.2f}'
                      '\n'
                      f'Kerf $w$ = {self.kerf*1e3:.3f} mm'
                      '\n'
                      f'Pitch $d$ = {self.pitch*1e3:.2f} mm'
                      f' = {self.p_lambda():.2f}'
                      r' $\lambda$'
                      )

        angle_text = (r'Steering angle $\theta_s$ = '
                      fr'{self.angle_s:.1f}$^\circ$')

        distance_text = (r'Rayleigh distance $z_R$ = '
                         f'{self.z_r():.2f} m')

        result_text = header + '\n' + array_text + '\n' + angle_text + \
            '\n' + distance_text

        ca.remove_fig_text(self.fig)
        ca.set_fig_text(self.fig, result_text, xpos=0.02, ypos=0.35)

        return

    def _initialise_graphs(self):
        """Initialise result graphs."""
        plt.close('all')
        fig = plt.figure(figsize=[15, 5],
                         constrained_layout=True,
                         num='Array Beamprofile')
        ca.add_logo(fig)

        gs = GridSpec(2, 4, figure=fig)
        ax = {'axial': fig.add_subplot(gs[0:, 2:]),
              'delay': fig.add_subplot(gs[0, 1]),
              'beamprofile': fig.add_subplot(gs[1, 1])}

        # Axial intensity plot
        ax['axial'].set(aspect='equal',
                        ylabel='Depth (z) [m]',
                        xlabel='Azimuth (x) [m]',
                        facecolor=self.intensity_background)
        # ax["axial"].yaxis.set_inverted(True)

        # Delay graph
        ax['delay'].set(xlabel='Element no.',
                        ylabel=r'Delay [$\mu$s]')

        ax['delay'].grid(visible=True, which='major')

        # Beam profile graph
        ax['beamprofile'].set(xlabel=r'Angle $\theta$ [$^\circ$]',
                              ylabel=r'$D(\theta)$ [dB re. max]')

        ax['beamprofile'].grid(visible=True, which='both', axis='x')

        return ax, fig

    def _remove_old_artists(self):
        for ax in self.axes.values():
            ca.remove_artists(ax)

        try:
            self.cbar.remove()
        except Exception:
            pass

        return 0

    # Interactive widgets
    def _create_widgets(self):
        """Create widgets for interactive operation."""
        title = 'Beam-profile from Single Element Transducer'
        title_widget = widgets.Label(title, style=dict(font_weight='bold'))

        text_layout = {'continuous_update': False,
                       'layout': widgets.Layout(width='95%'),
                       'style': {'description_width': '50%'}}

        slider_layout = {'continuous_update': False,
                         'layout': widgets.Layout(width='95%'),
                         'style': {'description_width': '30%'}}

        text_width = '20%'
        slider_width = '60%'

        # Text widgets (Dropboxes, number boxes)
        n_elements_widget = widgets.BoundedIntText(
            value=64, min=1, max=1024, step=1,
            description='No. of elements',
            **text_layout)

        frequency_widget = widgets.BoundedFloatText(
            value=100, min=1, max=400, step=1,
            description='Frequency [kHz]',
            **text_layout)

        pitch_widget = widgets.BoundedFloatText(
            value=7.5, min=0.1, max=100, step=0.1,
            description='Element pitch [mm]',
            **text_layout)

        db_range_widget = widgets.BoundedFloatText(
            value=60, min=6, max=120, step=6,
            description='Range [dB]',
            **text_layout)

        db_gain_widget = widgets.BoundedFloatText(
            value=0, min=-120, max=120, step=6,
            description='Gain [dB]',
            **text_layout)

        col_par = widgets.VBox([n_elements_widget,
                                frequency_widget,
                                pitch_widget],
                               layout=widgets.Layout(width=text_width))

        col_db = widgets.VBox([db_range_widget,
                              db_gain_widget],
                              layout=widgets.Layout(width=text_width))

        # Slider widgets
        steering_angle_widget = widgets.FloatSlider(
            min=-90, max=90,
            value=0, step=1,
            readout_format='.0f',
            description='Steering angle [Deg.]',
            **slider_layout)

        col_slider = widgets.VBox([steering_angle_widget],
                                  layout=widgets.Layout(width=slider_width))

        widget_layout = widgets.HBox([col_par, col_db, col_slider],
                                     layout=widgets.Layout(width='80%'))

        widget_layout = widgets.VBox([title_widget, widget_layout])

        widget = {'n_elements': n_elements_widget,
                  'frequency': frequency_widget,
                  'pitch': pitch_widget,
                  'db_range': db_range_widget,
                  'db_gain': db_gain_widget,
                  'steering_angle': steering_angle_widget,
                  }

        w = WidgetLayout(widget_layout, widget)

        return w
