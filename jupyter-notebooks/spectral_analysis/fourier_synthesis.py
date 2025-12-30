"""Demonstrate Fourier synthesis: Build waveform from cosine-waves."""
# Import libraries

from math import pi
import numpy as np
from scipy.fft import fft, fftshift, fftfreq
from scipy.signal import square, sawtooth
import matplotlib.pyplot as plt
import ipywidgets as widgets


class WidgetLayout():
    """Container for widgets and layout."""

    def __init__(self, layout, widget):
        self.layout = layout
        self.widget = widget


class FourierSynthesis():
    """Demonstation of Fourier synthesis.

    All calculations and plotting routines are contained in this class
    """

    def __init__(self, initialise_graphs=True, create_widgets=True):
        """Initialise signal."""
        self.waveform = 'square'  # Waveform type
        self.phase = 0            # Phase [degrees]
        self.offset = 0           # DC offset
        self.duty_cycle = 50      # Waveform duty cycle [%]
        self.n_coeff = 4          # Number of coefficients to show
        self.n_cycles = 2         # No. of cycles to plot
        self.n_colors = 10
        self.n_samples = 1024

        self.waveform_options = {0: 'Square',
                                 1: 'Triangle',
                                 2: 'Sawtooth',
                                 3: 'Cosine'}

        self.waveform = self.waveform_options[0]

        if initialise_graphs:
            self.color = self._initialise_colors()
            self.ax_time, self.ax_freq = self._initialise_graphs()

        if create_widgets:
            self.widget = self._create_widgets()

    def t(self):
        """Create time vector, normalised, from 0 to 2*pi."""
        # Time-vector with extra point, rmove end-point for exactly one cycle
        t = np.linspace(0, 2*pi, self.n_samples+1)
        t = t[0:-1]
        return t

    def s(self):
        """Create signal array."""
        phi = np.radians(self.phase)
        if self.waveform == self.waveform_options[0]:
                s = square(self.t() + phi, duty=self.duty_cycle/100)
        elif self.waveform == self.waveform_options[1]:
                s = sawtooth(self.t() + pi/2 + phi, width=0.5)
        elif self.waveform == self.waveform_options[2]:
                s = sawtooth(self.t() + phi)
        elif self.waveform == self.waveform_options[3]:
                s = np.cos(self.t() + phi)
        else:
            print('Unknown waveform')
            s = np.zeros_like(self.t())

        s = s + self.offset
        return s

    def fourier_coefficients(self):
        """Fourier coefficients."""
        a = fft(self.s())/self.n_samples  # Fourier coefficients, scaled
        f = fftfreq(self.n_samples, 1/self.n_samples)  # Frequency vector
        f = fftshift(f)                 # Shift negative frequencies to start
        a = fftshift(a)

        nz = np.argwhere(abs(a) < 1e-6)  # Set very small coefficients to zero 
        a[nz] = 0

        return a, f

    def partial_waves(self):
        """Calculate waves to synthesize signal."""
        k_max = self.n_coeff + 1    # First coefficient is zero
        a, f = self.fourier_coefficients()
        sp = np.zeros((self.n_samples, k_max))
        for k in range(k_max):
            k_twosided = np.argwhere(abs(f) == k)
            x = 0
            for m in k_twosided:
                # Sum of positive and nagative exponential
                x = x + a[m] * np.exp(1j*f[m]*self.t())
            # Remove imaginary part from roundoff error
            sp[:, k] = np.real_if_close(x)

        ss = np.sum(sp, axis=1)

        return sp, ss

    def display(self):
        """Plot all signals and spectra."""
        # Clear old lines
        for ax in self.ax_time + self.ax_freq:
            for art in list(ax.lines):
                art.remove()
            for art in list(ax.collections):
                art.remove()
            for art in list(ax.patches):
                art.remove()

            ax.axhline(y=0, color='gray')

        for ax in self.ax_freq:
            ax.axvline(x=0, color='gray')

        # Plot time traces
        s = np.append(self.s(), self.s())   # Display two cycles
        t = np.append(self.t(), self.t()+2*pi)

        sp, ss = self.partial_waves()
        sp = np.append(sp, sp, axis=0)
        ss = np.append(ss, ss, axis=0)

        # Original signal
        self.ax_time[0].plot(t, s,
                             linestyle='-',
                             color=self.color['original'])

        # Waves from Fourier components
        for k in range(self.n_coeff+1):
            self.ax_time[1].plot(
                t, sp[:, k],
                linestyle='-',
                color=self.color['partial'][k % self.n_colors])

        # Reconstructed signal
        self.ax_time[2].plot(t, ss,
                             linestyle='-',
                             color=self.color['original'])

        s_max = 1.4 * np.max(s)
        for k in range(len(self.ax_time)):
            self.ax_time[k].set(ylim=(-s_max, s_max))

        # Fourier coefficients
        a, f = self.fourier_coefficients()
        kp = np.argwhere(abs(f) < self.n_coeff+1)

        self.ax_freq[0].stem(f[kp], np.abs(a[kp]))
        self.ax_freq[1].stem(f[kp], np.angle(a[kp])/pi)
        self.ax_freq[0].set_ylim(0, s_max/2)

        return 0

    # Simple interactive operation
    def interact(self,
                 waveform=None,
                 phase=None,
                 offset=None,
                 duty_cycle=None,
                 n_coeff=None):
        """Scale inputs and  display results.

        For interactive operation.
        Existing values are used if a parameter is omitted.

        Parameters
        ----------
        waveform: float, optional
            Carrier frequency
        phase: float, optional
            Phase in degrees
        offset: float, optional
            DC offset
        n_coeff: int, optional
            No. of Fourier coefficients to include
        """
        if waveform is not None:
            self.waveform = waveform
        if phase is not None:
            self.phase = phase
        if offset is not None:
            self.offset = offset
        if duty_cycle is not None:
            self.duty_cycle = duty_cycle
        if n_coeff is not None:
            self.n_coeff = n_coeff

        # Display result in graphs
        self.display()

        return

    def _initialise_colors(self):
        """Create consistent set of colors for the plots."""
        color = {}
        color['original'] = 'C0'
        color['partial'] = ['C'+str(k) for k in range(self.n_colors)]

        return color

    def _initialise_graphs(self):
        """Initialise graphs for signals and spectra."""
        plt.close('all')
        plt.rc('font', size=9)          # Default text sizes
        fig = plt.figure(figsize=[12, 6],
                         constrained_layout=True,
                         num='Fourier Synthesis')

        n_plots = 3
        ax_time = [fig.add_subplot(n_plots, 2, 2*k+1) for k in range(n_plots)]
        ax_freq = [fig.add_subplot(n_plots, 2, 2*k+4) for k in range(2)]

        ax_time[0].set_title('Original signal')
        ax_time[1].set_title('Harmonic components')
        ax_time[2].set_title('Reconstructed signal')
        ax_freq[0].set_title('Magnitude')
        ax_freq[1].set_title(r'Phase [$\pi$ radians]')

        s_max = 1.5  # Max. amplitude on time domain plots
        for k in range(n_plots):
            ax_time[k].set(xlim=(0, 4*pi),
                           ylim=(-s_max, s_max))

        f_max = 20
        for k in range(2):
            ax_freq[k].set(xlim=(-f_max, f_max))
        ax_freq[0].set_ylim(0, s_max/2)
        ax_freq[1].set_ylim(-1, 1)

        return ax_time, ax_freq

    # Interactive widgets
    def _create_widgets(self):
        """Create widgets for interactive operation."""
        # Title
        title = ('Fourier Synthesis')
        title_widget = widgets.Label(title, style=dict(font_weight='bold'))

        # Layouts definitions
        text_layout = {'continuous_update': False,
                       'layout': widgets.Layout(width='95%'),
                       'style': {'description_width': '70%'}}

        dropdown_layout = {'layout': widgets.Layout(width='95%'),
                           'style': {'description_width': '50%'}}

        slider_layout = {'continuous_update': True,
                         'layout': widgets.Layout(width='90%'),
                         'style': {'description_width': '10%'}}

        # Individual widgets
        phase_widget = widgets.FloatText(
            min=-360.0, max=360.0, value=self.phase, step=5,
            description='Phase [deg]',
            readout_format='.0f',
            **text_layout)

        offset_widget = widgets.FloatText(
            min=-1.0, max=1.0, value=self.phase, step=0.1,
            description='DC offset',
            readout_format='.2f',
            **text_layout)

        n_coeff_widget = widgets.IntText(
            min=0, max=30, value=self.n_coeff, step=1,
            description='No. of coefficients',
            **text_layout)

        duty_cycle_widget = widgets.FloatText(
            min=5, max=95, value=self.duty_cycle, step=5,
            description='Duty cycle [%]',
            readout_format='.0f',
            **text_layout)

        waveform_widget = widgets.Dropdown(
            options=[*self.waveform_options.values()],
            value=self.waveform_options[0],
            description='Waveform Type',
            **dropdown_layout)

        # Arrange in columns and lines
        widget_col_1 = widgets.VBox([waveform_widget,
                                     duty_cycle_widget],
                                    layout=widgets.Layout(width='95%'))

        widget_col_2 = widgets.VBox([phase_widget,
                                     offset_widget],
                                    layout=widgets.Layout(width='95%'))

        widget_col_3 = widgets.VBox([n_coeff_widget],
                                    layout=widgets.Layout(width='95%'))

        widget_layout = widgets.HBox([widget_col_1,
                                      widget_col_2,
                                      widget_col_3,
                                      ],
                                     layout=widgets.Layout(width='40%'))

        widget_layout = widgets.VBox([title_widget, widget_layout])

        # Export as dictionary
        widget = {'waveform': waveform_widget,
                  'offset': offset_widget,
                  'phase': phase_widget,
                  'duty_cycle': duty_cycle_widget,
                  'n_coeff': n_coeff_widget,
                  }

        w = WidgetLayout(widget_layout, widget)

        return w
