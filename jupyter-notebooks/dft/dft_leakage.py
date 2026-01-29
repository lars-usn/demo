"""Demonstrate Fourier synthesis: Build waveform from cosine-waves."""
# Import libraries

from math import pi
import numpy as np
import matplotlib.pyplot as plt
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
        self.n_samples = 16   # Number of coefficients in DFT
        self.a = 1.0          # Amplitude
        self.phase = 0        # Phase [radians]
        self.k = 2.3          # Incoming frequency [rel DFT bin]
        self.m = 2            # Bin to show
        self.show_dtft = False   # Display DTFT spectrum

        if initialise_graphs:
            self.ax_time, self.ax_dft = self._initialise_graphs()

        if create_widgets:
            self.widget = self._create_widgets()

    def n(self):
        """Create time vector, samples."""
        return np.arange(self.n_samples)

    def ni(self):
        """Create time vector, interpolated between samples."""
        return np.arange(0, self.n_samples, 1/100)

    def x_in(self, n):
        """Create incoming signal as array."""
        s = self.a * np.cos(2*pi*self.k * n/self.n_samples + self.phase)
        return s

    def partial_waves(self, n):
        """Calculate interpolated sinusoids for DFT bins."""
        # Use lower half of spectrum, unique part
        m_max = int(np.ceil((self.n_samples/2)))

        X = self.dft()
        xc = [X[m] * np.exp(2j*pi * n * m / self.n_samples)
              for m in np.arange(m_max+1)]
        x = 2/self.n_samples * np.real(xc)

        return x

    def dft(self):
        """Calculate DFT."""
        x_i = self.x_in(self.n())
        n = self.n()

        ma = np.arange(self.n_samples)
        X = np.zeros_like(ma, dtype=np.complex64)

        for m in ma:
            e = np.exp(-2j*pi*n*m / self.n_samples)
            X[m] = np.sum(x_i * e, axis=0)

        X[np.where(np.abs(X) < 1e-8)] = 0

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
        """Discrete time Fourier transform for frequency k.

        Using the analytical solution in the frequency domain,
        the  Dirichlet-kernel
        """
        m = self.ni()
        N = self.n_samples
        kr = [self.k, -self.k]     # Include negative frequency

        Xi = [np.exp(-1j*pi*(N-1)/N*(m-k)) * self.dirichlet(k-m, N)
              for k in kr]

        X = 1/2 * self.a * np.sum(np.array(Xi), axis=0)

        return X, m

    def display(self):
        """Plot all signals and spectra."""
        incoming_color = 'C1'
        marker_color = 'C2'

        for ax in self.ax_time + self.ax_dft:
            # Clear old lines
            for art in list(ax.lines):
                art.remove()
            for art in list(ax.collections):
                art.remove()
            for art in list(ax.patches):
                art.remove()

            # Add common lines
            ax.axhline(y=0, color='gray')

        # Incoming signal, time domain
        x_i = self.x_in(self.n())
        xi_i = self.x_in(self.ni())

        self.ax_time[0].plot(self.ni(), xi_i, '-',
                             color=incoming_color,
                             label='Incoming signal')
        self.ax_time[0].plot(self.n(), x_i, 'o',
                             color=incoming_color)

        # Reconstructed signals from DFT
        xp = self.partial_waves(self.ni())
        self.ax_time[0].plot(self.ni(), np.transpose(xp), ':', color='C0')
        self.ax_time[0].plot(self.ni(), xp[self.m], '-', color=marker_color,
                             label='Selected DFT bin')

        # DFT bins
        X = self.dft()
        self.ax_dft[0].stem(np.abs(X), basefmt='gray')
        self.ax_dft[1].stem(np.degrees(np.angle(X)), basefmt='gray')

        # Mark reference bin
        for k in range(2):
            self.ax_dft[k].axvline(x=self.m,
                                   linestyle='--',
                                   color=marker_color)

        # DTFT curve as background for DFT
        if self.show_dtft:
            X, mi = self.dtft()
            dtft_curve = {'linestyle': '--',
                          'color': incoming_color,
                          'label': 'DTFT, continous'}

            self.ax_dft[0].plot(mi, np.abs(X), **dtft_curve)
            self.ax_dft[1].plot(mi, np.degrees(np.angle(X)), **dtft_curve)
            self.ax_time[0].legend()

        # Legends for all plots

        self.ax_dft[0].legend()
        return 0

    # Simple interactive operation
    def interact(self, k=None, m=None, show_dtft=None):
        """Scale inputs and  display results.

        For interactive operation.
        Existing values are used if a parameter is omitted.

        Parameters
        ----------
        k : float, optional
            Frequency of incoming wave, rel. DFT component number
        m : int, optional
            DFT component to show
        show_dtft  : int, optional
            Display DTFT spectrum
        """
        if k is not None:
            self.k = k
        if m is not None:
            self.m = m
        if show_dtft is not None:
            self.show_dtft = show_dtft
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

        ax_time = [fig.add_subplot(2, 2, 1)]
        ax_dft = [fig.add_subplot(2, 2, 2*k+2) for k in range(2)]

        ax_time[0].set_title('Incoming signal with DFT waves')
        ax_dft[0].set_title('DFT magnitude')
        ax_dft[1].set_title('DFT phase')

        n_range = [-0.5, self.n_samples-0.5]
        for ax in ax_time:
            ax.set(xlabel='n',
                   xlim=n_range)

        for ax in ax_dft:
            ax.set(xlabel='m',
                   xlim=n_range)

        ax_dft[1].set(ylim=(-180, 180),
                      yticks=range(-180, 181, 45))

        ax_dft[0].set_ylabel('Magnitude')
        ax_dft[1].set_ylabel('Phase [deg]')

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

        dropdown_layout = {
            'layout': widgets.Layout(width='90%'),
            'style': {'description_width': '40%'}}

        # Individual widgets
        m_widget = widgets.IntSlider(
            min=0, max=self.n_samples/2, value=self.m,
            description='Selected coefficient  $m$',
            **slider_layout)

        k_widget = widgets.FloatSlider(
            min=0, max=self.n_samples/2, value=self.k, step=0.1,
            description='Incoming frequency',
            readout_format='.1f',
            **slider_layout)

        show_dtft_widget = widgets.Checkbox(
            value=False,
            description='Display DTFT spectrum')

        widget_line = widgets.HBox([k_widget, show_dtft_widget])
        widget_layout = widgets.VBox([title_widget,
                                      widget_line,
                                      m_widget])

        # Export as dictionary
        widget = {'m': m_widget,
                  'k': k_widget,
                  'show_dtft': show_dtft_widget}

        w = WidgetLayout(widget_layout, widget)

        return w
