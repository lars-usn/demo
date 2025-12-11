"""Demonstrate Processing gain vs. DFT bins."""
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
        # Incoming signal
        self.f = 20    # Frequency
        self.a = 1.0     # Signal amplitude
        self.phase = 0.0  # Incoming signal phases [radians]
        self.noise = 1.0  # Noise amplitude (Gaussian std. dev.)

        self.n_win = 512          # DFT length
        self.n_win_list = [2**k for k in range(6, 15)]

        self.m_max = 100

        if initialise_graphs:
            self.ax_time, self.ax_dft = self._initialise_graphs()
            self.scale_graphs()

        if create_widgets:
            self.widget = self._create_widgets()

    def n(self):
        """Create time-domain signal."""
        return np.arange(self.n_win)

    def m_peak(self):
        """DFT bin corrspeonfing to signal frequency."""
        if self.f>=1:
            m = int(round(self.f))   # Interpret f as cycles/sample
        else:
            m = int(round(self.f * self.n_win)) # Interpret f as cycles/window

        return m

    def x_in(self):
        """Create time-domain signal, assumed infinite duration."""
        w = 2*pi*self.f    # Interpret f as cycles/sample
        if self.f>=1:
            w = w / self.n_win # Interpret f as cycles/window

        x = self.a * np.cos(w * self.n() + self.phase)

        rng = np.random.default_rng()
        noise = rng.normal(0.0, self.noise, self.n_win)

        return [x, noise, x+noise]

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

        w = signal.get_window('hann', self.n_win)

        m = fftfreq(n_fft, d=1/n_signal)
        m = fftshift(m)
        X = scale * 2 * fft(w*x, n_fft) / n_signal
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
            for art in list(ax.texts):
                art.remove()

        for ax in self.ax_time:
            ax.axhline(y=0, color='gray')

        # Signals and spectra
        n = self.n()
        x_i = self.x_in()

        spec = [self.spectrum(x_i[k]) for k in range(3)]
        X_db = [spec[k][0] for k in range(3)]
        m = spec[0][2]

        # Incoming signal
        for k in range(3):
            self.ax_time[k].plot(n, x_i[k], '-', color='C0')
            self.ax_dft[k].plot(m, X_db[k], 'C0')

        # Indicate SNR
        X_sdb = X_db[2]
        noise_level = np.percentile(X_sdb, 90)

        m_peak = self.m_peak()
        idx_peak = np.argwhere(m==m_peak)[0][0]
        X_peak = X_sdb[idx_peak]
        snr = X_peak - noise_level

        m_snr = self.m_max - 10

        self.ax_dft[2].axhline(y=noise_level, color='C3')
        self.ax_dft[2].plot(m_snr*np.ones(2), [noise_level, X_peak],
                            color='C3', linestyle=':', marker='_')

        annotationpar = {'textcoords': 'offset fontsize',
                         'horizontalalignment': 'left',
                         'verticalalignment': 'bottom',
                         'color': 'C3',
                         'backgroundcolor': 'white'}

        self.ax_dft[2].annotate(f'{noise_level:.1f} dB',
                                (0.7*self.m_max, noise_level),
                                (0, 0),
                                **annotationpar)

        self.ax_dft[2].annotate(f'{X_peak:.1f} dB',
                                (1.0*m_peak, X_peak),
                                (2, 0),
                                **annotationpar)

        self.ax_dft[2].annotate(f'SNR = {snr:.1f} dB',
                                (m_snr, X_peak),
                                (1, 0),
                                **annotationpar)
        self.scale_graphs()

        return 0

    def scale_graphs(self):
        """Scale graph axes."""
        for ax in self.ax_time:
            ax.set(ylim=(-4, 4),
                   xlim=(0, self.n_win))

        for ax in self.ax_dft:
            ax.set(ylim=(-42, 0),
                   xlim=(0, self.m_max))

        return 0

    # Simple interactive operation

    def interact(self, n_win=None, a=None, f=None):
        """Scale inputs and  display results.

        For interactive operation.
        Existing values are used if a parameter is omitted.

        Have not found a method to transfer array values as widgets.

        """
        if n_win is not None:
            self.n_win = int(n_win)
        if a is not None:
            self.a = a
        if f is not None:
            self.f = f
        self.display()

        return

    def _initialise_graphs(self):
        """Initialise graphs for signals and spectra."""
        plt.close('all')
        plt.rc('font', size=9)          # Default text sizes
        fig = plt.figure(figsize=[12, 6],
                         layout='constrained',
                         num='DFT')

        gs = GridSpec(3, 2, figure=fig)
        ax_time = [fig.add_subplot(gs[k, 0]) for k in range(3)]
        ax_dft = [fig.add_subplot(gs[k, 1]) for k in range(3)]

        ax_time[0].set_title('Sinusoid, single frequency')
        ax_time[1].set_title('Random noise')
        ax_time[2].set_title('Sinusoid with noise')

        ax_dft[0].set_title('Spectrum of sinusoid')
        ax_dft[1].set_title('Spectrum of noise')
        ax_dft[2].set_title('Spectrum of sinusoid and noise')

        for ax in ax_time:
            ax.set(xlabel='n')

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
        title = ('DFT processing gain')
        title_widget = widgets.Label(title, style=dict(font_weight='bold'))

        # Layouts definitions
        slider_layout = {'continuous_update': True,
                         'layout': widgets.Layout(width='40%'),
                         'style': {'description_width': '20%'}}

        # Individual widgets
        text_layout = {'continuous_update': False,
                       'layout': widgets.Layout(width='15%'),
                       'style': {'description_width': '50%'}}

        n_win_widget = widgets.FloatLogSlider(
            base=2,
            min=7, max=14, value=self.n_win, step=1,
            description='DFT length',
            readout_format='.0f',
            **slider_layout)

        a_widget = widgets.FloatSlider(
            min=0, max=2, value=self.a, step=0.1,
            description='Sinusoid amplitude',
            readout_format='.1f',
            **slider_layout)

        f_widget = widgets.FloatSlider(
            min=1, max=self.m_max, value=self.f, step=0.1,
            description='Sinusoid frequency',
            readout_format='.1f',
            **slider_layout)

        widget_layout = widgets.VBox([title_widget, n_win_widget, a_widget, f_widget])

        # Export as dictionary
        widget = {'n_win': n_win_widget,
                  'a': a_widget,
                  'f': f_widget,
                  }
        w = WidgetLayout(widget_layout, widget)

        return w
