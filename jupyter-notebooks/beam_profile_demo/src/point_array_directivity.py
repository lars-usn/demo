"""Calculate the directivity from an arrai of point sources

An interactive version can be run from the Jupyter Notebook 'point_array
_demo.ipynb'
"""

# Python libraries
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import ipywidgets as widgets

# Internal libraries
import beamplot_utilities as bpu


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
        self.n_elements = 16  # Number of elements in array
        self.pitch = 7.5e-3     # m   Pitch between elements
        self.frequency = 100e3  # Hz  Ultrasound frequency
        self.angle_s = 20       # deg Steering angle
        self.c = 1500           # m/s Speed of soundin load medium

        # Calculation settings
        self.z_ref = 20.0        # m    Reference depth
        self.y_lim = 0.5         # Relative limit for beamwidth

        # Display settings
        self.theta_max = 90    # deg  Max. angle to calculate
        self.x_max = 70.0      # m    Max. lateral dimension to calculate
        self.z_max = 140.0     # m    Max. depth to calculate
        self.n_x = 301  # No. of points in the x-direction (azimuth)
        self.n_z = 400  # No. of points in the z-direction (depth)
        self.db_range = 60     # dB   Dynamic gange on dB-scales
        self.db_gain = 6       # dB   Max. on dB-scales
        self.db_polar = 24     # dB   Dynamic range on polar plot

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
        """Steering angle in radians.."""
        return np.radians(self.angle_s)

    def delay(self):
        """Delay between elements (no focusing)."""
        return -np.sin(self.theta_s()) * self.pitch / self.c

    def delay_array(self):
        """Delay over all elements."""
        nc = 1/2*(self.n_elements+1)

        n = np.arange(0, self.n_elements)+1
        tau = -(n-nc) * self.delay()
        tau = tau
        return [tau, n]

    def wavelength(self):
        """Calculate acoustic wavelenght."""
        return self.c/self.frequency

    def d_aperture(self):
        """Width of aperture."""
        return self.n_elements * self.pitch

    def p_lambda(self):
        """Pitch relative to wavelength."""
        return self.pitch / self.wavelength()

    def d_lambda(self):
        """Aperture width relative to wavelength."""
        return self.d_aperture() / self.wavelength()

    def directivity(self, theta):
        """Directivity of array of points."""
        s_theta = np.sin(theta) - np.sin(self.theta_s())
        x = pi * self.p_lambda() * s_theta
        x[x == 0] = 1e-4   # Avoid 0/0 erroers
        d = np.sin(self.n_elements * x) / (self.n_elements * np.sin(x))

        return d

    # === Commands =============================
    def display(self):
        """Display beam pattern in graphs."""
        self._remove_old_artists()
        ax = self.axes

        # Delay profile
        tau, n = self.delay_array()
        ax['delay'].stem(n, tau*1e6, 'C0')

        # Directivity, polar plot
        theta = np.linspace(-pi/2, pi/2, 701)
        p_array = self.directivity(theta)

        ax['directivity'].plot(theta, bpu.db(p_array, p_ref=1),
                               color='C0', linestyle='solid')

        self.scale_axes()
        self._resulttext()
        return

    def scale_axes(self):
        """Change scales of all graphs."""
        ax = self.axes

        delay_max = 1e6 * 1/2 * self.n_elements * self.pitch / self.c
        ax['delay'].set(xlim=(1, self.n_elements),
                        ylim=(-delay_max, delay_max))
        bin_log = np.floor(np.log2(self.n_elements))
        dn = 2**(bin_log-2)

        ax['delay'].set_xticks(np.arange(1, self.n_elements), minor=True)
        ax['delay'].set_xticks(np.arange(0, self.n_elements+1, dn))

        ax['directivity'].set(thetamin=-90,
                              thetamax=+90,
                              theta_zero_location='S',
                              rmin=-self.db_polar,
                              rmax=0.0,
                              rticks=np.flip(np.arange(0, -self.db_polar, -6)))

        ax['directivity'].set_title('Radiation Diagram [dB re. max]', y=1)

        return 0

    def interact(self,
                 n_elements=None,
                 frequency=None,
                 pitch=None,
                 angle_s=None,
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
                      f'Pitch $d$ = {self.pitch*1e3:.2f} mm'
                      f' = {self.p_lambda():.2f}'
                      r' $\lambda$'
                      '\n'
                      f'Array width $D$ = {self.d_aperture()*1e3:.0f} mm'
                      f' = {self.d_lambda():.1f}'
                      r' $\lambda$'
                      )

        angle_text = (r'Steering angle $\theta_s$ = '
                      fr'{self.angle_s:.1f}$^\circ$')

        result_text = header + '\n' + array_text + \
            '\n' + angle_text

        bpu.set_fig_text(self.fig, result_text, xpos=0.75, ypos=0.10)

        return

    def _initialise_graphs(self):
        """Initialise result graphs."""
        plt.close('all')
        fig = plt.figure(figsize=[10, 5],
                         constrained_layout=True,
                         num='Array Beamprofile')
        bpu.add_logo(fig)

        gs = GridSpec(2, 4, figure=fig)
        ax = {'delay': fig.add_subplot(gs[0, 3]),
              'directivity': fig.add_subplot(gs[0:, 0:3], projection='polar')}

        # Delay graph
        ax['delay'].set(xlabel='Element no.',
                        ylabel=r'Delay [$\mu$s]')

        ax['delay'].grid(visible=True, which='major')

        return ax, fig

    def _remove_old_artists(self):
        for ax in self.axes.values():
            bpu.remove_artists(ax)

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

        slider_layout = {'continuous_update': True,
                         'layout': widgets.Layout(width='95%'),
                         'style': {'description_width': '30%'}}

        text_width = '30%'
        slider_width = '60%'

        # Text widgets (Dropboxes, number boxes)
        n_elements_widget = widgets.BoundedIntText(
            value=self.n_elements,
            min=1, max=1024, step=1,
            description='No. of elements',
            **text_layout)

        frequency_widget = widgets.BoundedFloatText(
            value=self.frequency/1e3,
            min=1, max=400, step=1,
            description='Frequency [kHz]',
            **text_layout)

        pitch_widget = widgets.BoundedFloatText(
            value=self.pitch*1e3,
            min=0.1, max=100, step=0.1,
            description='Element pitch [mm]',
            **text_layout)

        col_par = widgets.VBox([n_elements_widget,
                                frequency_widget,
                                pitch_widget],
                               layout=widgets.Layout(width=text_width))

        # Slider widgets
        steering_angle_widget = widgets.FloatSlider(
            value=self.angle_s,
            step=1,
            min=-90, max=90,
            readout_format='.0f',
            description='Steering angle [Deg.]',
            **slider_layout)

        col_slider = widgets.VBox([steering_angle_widget],
                                  layout=widgets.Layout(width=slider_width))

        widget_layout = widgets.HBox([col_par, col_slider],
                                     layout=widgets.Layout(width='80%'))

        widget_layout = widgets.VBox([title_widget, widget_layout])

        widget = {'n_elements': n_elements_widget,
                  'frequency': frequency_widget,
                  'pitch': pitch_widget,
                  'steering_angle': steering_angle_widget,
                  }

        w = WidgetLayout(widget_layout, widget)

        return w
