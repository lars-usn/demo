"""Demonstrate Fourier synthesis: Build waveform from cosine-waves."""
# Import libraries

from math import pi
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from scipy import signal
from scipy.fft import fft


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
        # Incoming single frequency wave
        self.a = 1.0        # Amplitude
        self.phase = 0      # Phase [radians]
        self.k = 10        # Incoming frequency [rel DFT bin]
        self.n_dft = 1024

        self.window_list = ['boxcar', 'triang', 'hamming', 'hann', 'tukey']
        self.window_name = self.window_list[0]

        # DFT bin to illustrate
        self.m = 1

        if initialise_graphs:
            self.ax_time, self.ax_dft = self._initialise_graphs()

        if create_widgets:
            self.widget = self._create_widgets()

    def n(self):
        """Create time vector, interpolated between samples."""
        p = self.n_dft / self.k        # Period, ref. sampling
        n_min = -2*self.n_dft
        n_l0 = 8*self.n_dft
        n_length = int(np.ceil(n_l0 / p) * p)

        return n_min + np.arange(n_length)

    def i_w(self):
        """Get indices of dft range."""
        i0, = np.nonzero(self.n() == 0)
        return i0 + np.arange(self.n_dft)

    def x_in(self):
        """Create incoming signal as array."""
        s = self.a * np.cos(2*pi*self.k * self.n()/self.n_dft + self.phase)
        return s

    def x_w(self):
        """Find signal inside window."""
        idx = self.i_w()
        return self.x_in()[idx]

    def window(self, window):
        """Select window function from name."""
        w = signal.get_window(window, self.n_dft)
        return w

    def dft(self, x):
        """Calculate DFT."""
        n = np.arange(len(x))
        ma = np.arange(len(x))
        X = np.zeros_like(ma, dtype=np.complex64)

        for m in ma:
            e = np.exp(-2j*pi*n*m / self.n_dft)
            X[m] = np.sum(x * e, axis=0)

        X[np.where(np.abs(X) < 1e-6)] = 0

        return X

    def dirichlet(self, x, N):
        """Dirichlet kernel."""
        nr = np.sin(pi*x)
        dr = np.sin(pi*x/N)
        d = np.divide(nr, dr,
                      out=np.full_like(nr, N),
                      where=dr != 0)

        return d

    def dtft(self):
        """Discrete time Fourier transform for frequency k."""
        m = self.ni()
        N = self.n_dft
        kr = [self.k, -self.k]

        Xi = [np.exp(1j*pi*(1-1/N)*(k-m)) * self.dirichlet(k-m, N)
              for k in kr]

        X = 1/2 * self.a * np.sum(np.array(Xi), axis=0)

        return X, m

    def display(self):
        """Plot all signals and spectra."""
        marker_color = 'C3'

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

        # Incoming signal
        self.ax_time[0].plot(n, x_i, '-', color='C0',
                             label='Incoming signal')
        self.ax_time[0].axvspan(0, self.n_dft, alpha=0.10, color='green')

        # Winowed signal
        win = self.window(self.window_name)
        xw = win * self.x_w()
        n_w = np.arange(self.n_dft)

        for ax in self.ax_time[0:2]:
            ax.plot(n_w, win, '-', color='C1',
                    label=self.window_name + ' window')

        self.ax_time[1].legend(loc='upper right')

        self.ax_time[2].plot(n_w, xw, '-', color='C0')

        # Spectra
        w = signal.get_window('hann', len(x_i))
        Xf = fft(x_i)/len(x_i)*2
        m = np.arange(len(Xf)) * self.n_dft / len(Xf)
        Xf_dB = 20*np.log10(np.abs(Xf))
        self.ax_dft[0].plot(m, Xf_dB, 'C0')

        # self.ax_time[0].plot(self.n(), x_i, 'o', color='C0')

        # DFT components
        # X = self.dft()
        # self.ax_dft[0].stem(np.abs(X), basefmt='gray')
        # self.ax_dft[1].stem(np.degrees(np.angle(X)), basefmt='gray')

        # for k in range(2):
        #     self.ax_dft[k].axvline(x=self.m,
        #                            linestyle='--', color=marker_color)

        # # DTFT curve as background for DFT components
        # X, mi = self.dtft()
        # self.ax_dft[0].plot(mi, np.abs(X), ':',
        #                     color='C0', label='DTFT, continous')
        # self.ax_dft[1].plot(mi, np.degrees(np.angle(X)), ':',
        #                     color='C0')

        # self.ax_time[0].legend()
        # self.ax_dft[0].legend()

        return 0

    # Simple interactive operation
    def interact(self, k=None, m=None):
        """Scale inputs and  display results.

        For interactive operation.
        Existing values are used if a parameter is omitted.

        Parameters
        ----------
        k : float, optional
            Frequency of incoming wave, rel. DFT component number
        m : int, optional
            DFT component to show
        """
        if k is not None:
            self.k = k
        if m is not None:
            self.m = m
        # Display result in graphs
        self.display()

        return

    def _initialise_graphs(self):
        """Initialise graphs for signals and spectra."""
        plt.close('all')
        plt.rc('font', size=9)          # Default text sizes
        fig = plt.figure(figsize=[12, 6],
                         layout='constrained',
                         num='DFT')

        ax_time = [fig.add_subplot(3, 2, 2*k+1) for k in range(3)]
        ax_dft = [fig.add_subplot(3, 2, 2*k+2) for k in range(3)]

        ax_time[0].set_title('Incoming signal - CW')
        ax_time[1].set_title('Window function')
        ax_time[2].set_title('Windowed signal')

        ax_dft[0].set_title('Spectrum of CW signal')
        ax_dft[1].set_title('Spectrum of window')
        ax_dft[2].set_title('Spectrum of windowed signal')

        for ax in ax_time:
            ax.set(xlabel='n')

        ax_time[0].set(xlim=[-1.5*self.n_dft, 3*self.n_dft])

        for ax in ax_dft:
            ax.set(xlabel='m',
                   ylabel='Magnitude [dB]',
                   ylim=(-40, 0),
                   xlim=(0, 20))

        return ax_time, ax_dft

    # Interactive widgets
    def _create_widgets(self):
        """Create widgets for interactive operation."""
        # Title
        title = ('DFT leakage')
        title_widget = widgets.Label(title, style=dict(font_weight='bold'))

        # Layouts definitions
        text_layout = {'continuous_update': False,
                       'layout': widgets.Layout(width='20%'),
                       'style': {'description_width': '70%'}}

        slider_layout = {'continuous_update': False,
                         'layout': widgets.Layout(width='60%'),
                         'style': {'description_width': '25%'}}

        # Individual widgets
        m_widget = widgets.IntSlider(
            min=0, max=self.n_dft/2, value=self.m,
            description='Selected coefficient  $m$',
            **slider_layout)

        k_widget = widgets.FloatSlider(
            min=0, max=self.n_dft/2, value=self.k, step=0.1,
            description='Incoming frequency',
            readout_format='.1f',
            **slider_layout)

        widget_layout = widgets.VBox([title_widget,
                                      k_widget,
                                      m_widget])

        # Export as dictionary
        widget = {'m': m_widget,
                  'k': k_widget}

        w = WidgetLayout(widget_layout, widget)

        return w
