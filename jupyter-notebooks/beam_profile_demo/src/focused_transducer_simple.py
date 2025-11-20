"""Run this to import libraries and define internal functions."""

# Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import ipywidgets as widgets

# Internal libraries
import beamplot_utilities as bpu


class WidgetLayout():
    """Container for widgets and layout."""

    def __init__(self, layout, widget):
        self.layout = layout
        self.widget = widget


class Transducer():
    """Define, calculate, and display transducer beam profile."""

    def __init__(self, create_widgets=False):

        # Transducer definition
        self.diameter = 20e-3   # m   Element diameter
        self.frequency = 3e6    # Hz  Ultrasound frequency
        self.focal_length = 50e-3    # m  Focal length

        self.c = 1540           # m/s Speed of soundin load medium

        # Display scale
        self.x_max = 40e-3
        self.z_max = 200e-3
        self.z_min = -3e-3

        # Colors and markers
        self.aperture_color = 'crimson'

        self.aperture_line = {'color': self.aperture_color,
                              'linestyle': 'solid',
                              'linewidth': 3}

        self.helper_line = {'color': 'darkgrey',
                            'linestyle': 'dashed'}

        self.marker_line = {'color': 'C1',
                            'linestyle': 'solid'}

        self.beam_line = {'color': 'C0',
                          'linestyle': 'solid'}

        self.focalzone_fill = {'color': 'C1',
                               'alpha': 0.7}

        self.baffle_fill = {'color': 'grey'}

        # Initialisation
        ax, fig = self._initialise_graphs()
        self.axes = ax
        self.fig = fig
        self.scale_axes()

        if create_widgets:
            self.widget = self._create_widgets()

    # === Calculated parameters ===========================
    def a(self):
        """Calculate transducer radius."""
        return self.diameter/2

    def wavelength(self):
        """Calculate acoustic wavelength."""
        return self.c/self.frequency

    def d_lambda(self):
        """Calculate aperture width relative to wavelength."""
        return self.diameter / self.wavelength()

    def theta_0(self):
        """Calculate opening angle from theory, two-sided, -12 dB."""
        return 2*self.wavelength()/self.diameter

    def f_number(self):
        """Calculate numeric aperture (f-number)."""
        return self.focal_length/self.diameter

    def z_r(self):
        """Calculate Rayleigh distance, far-field limit."""
        return self.diameter**2/(2*self.wavelength())

    def beamwidth(self):
        """Estimate beam width."""
        return self.theta_0()*self.focal_length

    def focalzone(self):
        """Find limits of the focal zone."""
        z1 = self.focal_length / (1 + self.theta_0()*self.f_number())
        x1 = z1*self.theta_0()/2

        z2 = self.focal_length / (1 - self.theta_0()*self.f_number())
        x2 = z2*self.theta_0()/2

        return [np.array([z1, z2]), np.array([x1, x2])]

    def focalzone_length(self):
        """Fnd length of focal zone."""
        z, x = self.focalzone()

        return z[1]-z[0]

    def focalzone_length_approx(self):
        """Calculate approximate length of focal zone."""
        return 4 * self.wavelength() * self.f_number()**2

    # === Calculated parameteres ===================

    def z(self):
        """Define depth-axis (z)."""
        return np.linspace(0, self.z_max, 300)

    def aperture_curve(self):
        """Calculate position of aperture."""
        phi_max = np.arcsin(self.a()/self.focal_length)
        phi = np.linspace(-phi_max, phi_max, 301)

        x = self.focal_length * np.sin(phi)
        z = self.focal_length * (1-np.cos(phi))

        return [z, x]

    def diffraction_curve(self):
        """Calculate outer beam profile from diffraction (opening angle)."""
        return self.z() * np.tan(self.theta_0()/2)

    def focusing_curve(self):
        """Calculate outer beam profile from focusing."""
        return self.a() * (1 - self.z() / self.focal_length)

    def beam_curve(self):
        """Estimate beam profile from combined opening angle and focusing."""
        return np.maximum(abs(self.diffraction_curve()),
                          abs(self.focusing_curve()))

    # === Commands =============================
    def display(self):
        """Display beam pattern in graphs."""
        self._remove_old_artists()
        ax = self.axes['beam']

        # Draw aperture and baffle
        z_a, x_a = self.aperture_curve()
        ax.plot(z_a*1e3, x_a*1e3, **self.aperture_line)

        xlim = ax.get_xlim()
        ax.axvspan(xmin=xlim[0], xmax=0, **self.baffle_fill)
        ax.fill_betweenx(y=x_a*1e3, x1=z_a*1e3, color=self.aperture_color)

        # Mark focal length and Rayleigh distances
        ax.axvline(x=self.focal_length*1e3, **self.marker_line)
        ax.axvline(x=self.z_r()*1e3, **self.helper_line)

        # Draw beam limits
        z = self.z()*1e3
        x_d = self.diffraction_curve() * 1e3
        x_f = self.focusing_curve() * 1e3
        x_b = self.beam_curve() * 1e3

        ax.plot(z, x_d, z, -x_d, **self.helper_line)
        ax.plot(z, x_f, z, -x_f, **self.helper_line)
        ax.plot(z, x_b, z, -x_b, **self.beam_line)

        # Mark focal zone
        z_fz, x_fz = self.focalzone()
        z_fz = np.concatenate((z_fz, np.flip(z_fz)))
        x_fz = np.concatenate((x_fz, np.flip(-x_fz)))
        ax.fill(z_fz*1e3, x_fz*1e3, **self.focalzone_fill)

        self._resulttext()

        return

    def scale_axes(self):
        """Change scales of all graphs."""
        ax = self.axes

        ax['beam'].set(ylim=self.x_max*np.array([-1, 1])*1e3,
                       xlim=np.array([self.z_min, self.z_max])*1e3)

        return 0

    def interact(self,
                 diameter=None,
                 frequency=None,
                 focal_length=None,
                 c=None,
                 ):
        """Scale inputs and  display results.

        For interactive operation with  dimensions in mm and frequency in kHz.
        Existing values are used if a parameter is omitted.

        Parameters
        ----------
        diameter: float, optional
            Element diameter in mm
        frequency: float, optional
            Frequency in MHz
        focal_length: float, optional
            Transducer focal length in mm
        c: float, optional
            Speed of sound in m/s
        """
        if diameter is not None:
            self.diameter = 1e-3*diameter
        if frequency is not None:
            self.frequency = 1e6*frequency
        if focal_length is not None:
            self.focal_length = 1e-3*focal_length
        if c is not None:
            self.c = c

        # Display result in graphs
        self.display()

        return

    # === Non-public methods ==========================================

    # Graphs and results
    def _resulttext(self):
        """Text box for lateral profile results."""
        z_f, x_f = self.focalzone()
        result_text = (f'Frequency  $f$ = {self.frequency/1e6:.1f} MHz'
                       '\n'
                       fr'Wavelength  $\lambda$ = '
                       fr'{self.wavelength()*1e3:.2f} mm'
                       '\n'
                       f'Diameter $D$ = {self.diameter*1e3:.0f} mm = '
                       fr'{self.d_lambda():.1f} $\lambda$'
                       '\n'
                       r'Focal length $F$ = '
                       f'{self.focal_length*1e3:.0f} mm'
                       '\n'
                       fr'F-number  $FN$ = '
                       f'{self.focal_length/self.diameter:.1f}'
                       '\n'
                       r'Rayleigh distance $z_R$ = '
                       f'{self.z_r()*1e3:.0f} mm'
                       '\n\n'
                       f'Opening angle, double-sided, -12 dB  '
                       r' $\theta_{-12dB}$ = '
                       fr'{np.degrees(self.theta_0()):.1f}$^\circ$'
                       '\n'
                       r'Beam width  $D_F$ = '
                       f'{self.beamwidth()*1e3:.1f} mm'
                       '\n'
                       'Focal zone '
                       '$z_{F1}$ = ' f'{z_f[0]*1e3:.1f} mm,    '
                       '$z_{F2}$ = ' f'{z_f[1]*1e3:.1f} mm'
                       '\n'
                       r'Focal zone length $L_F$ = '
                       f'{self.focalzone_length()*1e3:.1f} mm')

        bpu.remove_fig_text(self.fig)
        bpu.set_fig_text(self.fig, result_text, xpos=0.02, ypos=0.35)

        return

    def _initialise_graphs(self):
        """Initialise result graphs."""
        plt.close('all')
        fig = plt.figure(figsize=[12, 4],
                         constrained_layout=True,
                         num='Beam-Profile')
        bpu.add_logo(fig)

        gs = GridSpec(1, 5, figure=fig)
        ax = {'beam': fig.add_subplot(gs[0, 2:])}

        # Axial beam plot
        ax['beam'].set(aspect='equal',
                       xlabel='Depth (z) [mm]',
                       ylabel='Lateral (x) [mm]')
        ax['beam'].grid(visible='True', which='both')

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
        title = 'Beam-profile from Focused Transducer. Simple Estimate'
        title_widget = widgets.Label(title, style=dict(font_weight='bold'))

        left_layout = {'continuous_update': True,
                       'layout': widgets.Layout(width='95%'),
                       'style': {'description_width': '50%'}}

        right_layout = {'continuous_update': True,
                        'layout': widgets.Layout(width='95%'),
                        'style': {'description_width': '30%'}}

        left_width = '25%'
        right_width = '75%'

        # Left column widgets (Dropboxes, number boxes)
        soundspeed_widget = widgets.BoundedFloatText(
            value=1540, min=1000, max=2000, step=1,
            description='Speed of sound [m/s]',
            **left_layout)

        frequency_widget = widgets.BoundedFloatText(
            min=0.1, max=30.0, value=3.0, step=0.1,
            readout_format='3.1f',
            description='Frequency [MHz]',
            **left_layout)

        left_col = widgets.VBox([soundspeed_widget,
                                 frequency_widget],
                                layout=widgets.Layout(width=left_width))

        # Right column widgets (Sliders)
        diameter_widget = widgets.FloatSlider(
            min=1, max=50, value=20, step=1,
            readout_format='.0f',
            description='Diameter [mm]',
            **right_layout)

        focal_length_widget = widgets.FloatSlider(
            min=1, max=150, value=50, step=1,
            readout_format='.0f',
            description='Focal length [mm]',
            **right_layout)

        right_col = widgets.VBox([diameter_widget,
                                  focal_length_widget],
                                 layout=widgets.Layout(width=right_width))

        widget_layout = widgets.HBox([left_col, right_col],
                                     layout=widgets.Layout(width='80%'))

        widget_layout = widgets.VBox([title_widget, widget_layout])

        widget = {'diameter': diameter_widget,
                  'frequency': frequency_widget,
                  'focal_length': focal_length_widget,
                  'c': soundspeed_widget,
                  }

        w = WidgetLayout(widget_layout, widget)

        return w
