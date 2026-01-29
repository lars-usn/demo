"""Demonstrate Fourier synthesis: Build waveform from cosine-waves."""
# Import libraries

from math import pi
import numpy as np
from scipy import signal
from scipy.fft import fft, fftshift, fftfreq
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
import ipywidgets as widgets


class WidgetLayout():
    """Container for widgets and layout."""

    def __init__(self, layout, widget):
        self.layout = layout
        self.widget = widget


class Signal():
    """Demonstation of Fourier synthesis.

    All calculations and plotting routines are contained in this class
    """

    def __init__(self, initialise_graphs=True, create_widgets=False):
        """Initialise signal."""
        # Incoming signal, two frequencies
        self.f = np.array([15.0, 20.0])  # Incoming signal amplitudes
        self.a = np.array([1, 0.02])     # Incoming frequencies [rel DFT bin]
        self.phase = np.array([0.0, 0.0])    # Incoming signal phases [radians]

        self.n_win = 512                 # Length of sample window
        self.n_fft = 2**12        # Length of zero-padded signal
        self.n_fft_list = [0] + [2**k for k in range(8, 15)]

        self.window_list = ['boxcar', 'triang', 'hamming', 'hann', 'tukey']
        self.window_name = self.window_list[0]

        self.m_max = 40                # Max. bin to display

        if initialise_graphs:
            self.ax_time, self.ax_dft = self._initialise_graphs()
            self.scale_graphs()

        if create_widgets:
            self.widget = self._create_widgets()

    def n(self):
        """Create incoming time-domain vector, assumed infinite duration."""
        p = self.n_win / self.f[0]   # Samples/period of incoming signal
        n_min = int(-2*self.n_win)   # Integer no. of cycles for f[0]
        n_l0 = 8*self.n_win
        n_length = int(np.ceil(n_l0 / p) * p)

        return n_min + np.arange(n_length)

    def x_in(self):
        """Create incoming signal, assumed infinite duration."""
        x = 0
        for k in range(len(self.f)):
            x += self.a[k] * np.cos(2*pi*self.f[k] * self.n()/self.n_win
                                    + self.phase[k])
        return x

    def x_win(self):
        """Find signal inside window. Requires that n=0 exists."""
        i0, = np.nonzero(self.n() == 0)
        idx = i0 + np.arange(self.n_win)
        return self.x_in()[idx]

    def window(self):
        """Select window function from name."""
        w = signal.get_window(self.window_name, self.n_win)
        w[0] = w[-1] = 0    # Endpoints to zero for better illustration
        return w

    def spectrum(self, x, n_fft=0, scale=1.0):
        """Calculate spectrum of signal.

        Parameters
        ----------
        x: array of float
            signal, time-domain
        n_fft:int, optional
            Length of spectrum, zero-padded
        scale: float, optional
            Scaling factor for amplitude
        Returns
        -------
        m: Array of float
            Bin numbers rel. original signal length, shifted to zero center
        X: Array of complex
            DFT values
        x_dB: Array of float
            Magnitude of DFT in dB
        """
        n_signal = len(x)
        n_fft = max(n_signal, n_fft)

        m = fftfreq(n_fft, d=1/n_signal)
        m = fftshift(m)
        X = scale * 2 * fft(x, n_fft) / n_signal
        X = fftshift(X)

        Xa = np.clip(np.abs(X), 1e-40, None)  # Avoid log(zero)
        X_dB = 20*np.log10(Xa)

        return X_dB, X, m

    def display(self):
        """Plot all signals and spectra."""
        # Clear old lines
        for ax in self.ax_time + self.ax_dft:
            for art in list(ax.lines):
                art.remove()
            for art in list(ax.collections):
                art.remove()
            for art in list(ax.patches):
                art.remove()

        for ax in self.ax_time:
            ax.axhline(y=0, color='gray')

        # Incoming signals, time domain
        n = self.n()
        x_i = self.x_in()

        win = self.window()
        x_w = win * self.x_win()
        n_w = np.arange(self.n_win)

        # Incoming signal
        self.ax_time[0].plot(n, x_i, '-', color='C0')
        for n_lim in [0, self.n_win]:
            self.ax_time[0].axvline(x=n_lim, color='C1')
        # self.ax_time[0].axvspan(0, self.n_win, alpha=0.20, color='C1')

        self.ax_time[0].fill_between(n_w, win, -win, alpha=0.20, color='C1')

        # Winowed signal
        for ax in self.ax_time[0:2]:
            ax.plot(n_w, win, '-', color='C1',
                    label=self.window_name + ' window')

        self.ax_time[1].plot(n_w, x_w, '-', color='C0')

        self.ax_time[1].set_title(f'Windowed signal - {self.window_name} ')

        # Spectra
        X_db, X, m = self.spectrum(x_i)
        m = m * self.n_win / len(m)  # Rescale m to periods over window
        self.ax_dft[0].plot(m, X_db, 'C0')

        X_db, X, m = self.spectrum(win, self.n_fft, scale=0.5)
        self.ax_dft[1].plot(m, X_db, 'C1')
        self.ax_dft[1].axvline(x=0, color='gray')

        # Spectrum inside window
        X_db, X, m = self.spectrum(x_w, self.n_fft)
        self.ax_dft[2].plot(m, X_db, color='C0')

        self.scale_graphs()

        return 0

    def scale_graphs(self):
        """Scale graph axes."""
        self.ax_time[0].set(xlim=[-1.0*self.n_win, 2*self.n_win])

        for ax in self.ax_dft:
            ax.set(ylim=(-60, 0),
                   xlim=(0, self.m_max))

        self.ax_dft[1].set(xlim=(-self.m_max/2, self.m_max/2))
        return 0

    # Simple interactive operation

    def interact(self,
                 f_0=None, f_1=None,
                 a_0=None, a_1=None,
                 phase_0=None, phase_1=None,
                 window=None,
                 n_fft=None):
        """Scale inputs and  display results.

        For interactive operation.
        Existing values are used if a parameter is omitted.

        Have not found a method to transfer array values as widgets.

        """
        if a_0 is not None:
            self.a[0] = a_0
        if a_1 is not None:
            self.a[1] = a_1
        if f_0 is not None:
            self.f[0] = f_0
        if f_1 is not None:
            self.f[1] = f_1
        if phase_0 is not None:
            self.phase[0] = np.radians(phase_0)
        if phase_1 is not None:
            self.phase[1] = np.radians(phase_1)
        if window is not None:
            self.window_name = window
        if n_fft is not None:
            self.n_fft = n_fft

        self.display()

        return

    def _initialise_graphs(self):
        """Initialise graphs for signals and spectra."""
        plt.close('all')
        fig = plt.figure(figsize=[10, 5],
                         layout='constrained',
                         num='DFT')

        gs = GridSpec(2, 3, figure=fig)
        ax_time = [fig.add_subplot(gs[0, 0:2]),
                   fig.add_subplot(gs[1, 0])]

        ax_dft = [fig.add_subplot(gs[0, 2]),
                  fig.add_subplot(gs[1, 1]),
                  fig.add_subplot(gs[1, 2])]

        ax_time[0].set_title('Incoming signal - Infinite duration')
        ax_time[1].set_title('Windowed signal')
        ax_dft[0].set_title('Spectrum of incoming signal - Idealized')
        ax_dft[1].set_title('Spectrum of window')
        ax_dft[2].set_title('Spectrum of windowed signal')

        for ax in ax_time:
            ax.set(xlabel='n')

        ax_time[0].set(xlim=[-1.5*self.n_win, 3*self.n_win])

        for ax in ax_dft:
            ax.set(xlabel='m',
                   ylabel='Magnitude [dB]',
                   ylim=(-60, 6))

            ax.yaxis.set_major_locator(MultipleLocator(6))
            ax.yaxis.set_minor_locator(MultipleLocator(1))
            ax.grid(visible=True, which='major', axis='y')

        return ax_time, ax_dft

    # Interactive widgets
    def _create_widgets(self):
        """Create widgets for interactive operation."""
        # Title
        title = ('DFT windowing and zero-padding')
        title_widget = widgets.Label(title, style=dict(font_weight='bold'))

        # Layouts definitions
        text_layout = {'continuous_update': False,
                       'layout': widgets.Layout(width='15%'),
                       'style': {'description_width': '50%'}}

        slider_layout = {'continuous_update': True,
                         'layout': widgets.Layout(width='40%'),
                         'style': {'description_width': '20%'}}

        dropdown_layout = {
            'layout': widgets.Layout(width='15%'),
            'style': {'description_width': '50%'}}

        # Individual widgets
        a_widget = [widgets.FloatText(
            min=0, max=1, value=self.a[k], step=0.001,
            description=f'Amplitude {k}',
            readout_format='.3f',
            **text_layout)
            for k in range(2)]

        f_widget = [widgets.FloatSlider(
            min=1, max=self.m_max, value=self.f[k], step=0.1,
            description=f'Frequency {k}',
            readout_format='.1f',
            **slider_layout)
            for k in range(2)]

        phase_widget = [widgets.FloatText(
            min=-180.0, max=180.0, value=np.degrees(self.phase[k]), step=1.0,
            description=f'Phase [deg] {k}',
            readout_format='.0f',
            **text_layout)
            for k in range(2)]

        window_widget = widgets.Dropdown(
            options=self.window_list,
            value=self.window_list[0],
            description='Window',
            **dropdown_layout)

        n_fft_widget = widgets.Dropdown(
            options=self.n_fft_list,
            value=self.n_fft_list[4],
            description='DFT length',
            **dropdown_layout)

        signal_lines = [widgets.HBox([a_widget[k],
                                      f_widget[k],
                                      phase_widget[k]])
                        for k in range(2)]

        options_line = widgets.HBox([window_widget, n_fft_widget])

        widget_layout = widgets.VBox([title_widget]
                                     + signal_lines
                                     + [window_widget, n_fft_widget])

        # Export as dictionary
        widget = {'f_0': f_widget[0],
                  'f_1': f_widget[1],
                  'a_0': a_widget[0],
                  'a_1': a_widget[1],
                  'phase_0': phase_widget[0],
                  'phase_1': phase_widget[1],
                  'window': window_widget,
                  'n_fft': n_fft_widget}

        w = WidgetLayout(widget_layout, widget)

        return w
