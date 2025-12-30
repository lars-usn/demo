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

        self.a = np.array([1, 0.5])            # Phase [degrees]
        self.phase = np.array([0, 3/4*pi])            # Phase [degrees]
        self.f = np.array([1000, 2000])            # Phase [degrees]
        self.fs = 8000
        self.n_samples = 8          # Number of coefficients to show
        self.m = 1

        self.n_comp = len(self.a)

        if initialise_graphs:
            self.ax_time, self.ax_dft = self._initialise_graphs()

        if create_widgets:
            self.widget = self._create_widgets()

    def n(self):
        return np.arange(self.n_samples)

    def dt(self):
        return 1/self.fs

    def t_end(self):
        return (self.n_samples-1)/self.fs

    def ni(self):
        """Create time vector, normalised, from 0 to 2*pi."""
        # Time-vector
        return np.arange(0, self.n_samples, 1/40)

    def x(self, n):
        """Create signal array."""

        s = [self.a[k] * np.sin(2*pi*self.f[k]/self.fs * n +  self.phase[k])
             for k in range(self.n_comp)]

        s_sum = np.sum(s, axis=0)
        return s_sum, s

    def x_m(self, n):
        c = np.cos(2*pi* n * self.m /self.n_samples)
        s = -np.sin(2*pi* n * self.m /self.n_samples)

        return c, s


    def dft(self):
        """Fourier coefficients."""

        c, s = self.x_m(self.n())
        x_s, x_si = self.x(self.n())
        n = self.n()

        m= np.arange(self.n_samples)
        X = np.zeros_like(m, dtype=np.complex64)

        for mk in m:
            e = np.exp(-2j*pi* n * mk /self.n_samples)
            X[mk] = np.sum(x_s * e, axis=0)

        X[np.where(np.abs(X)<1e-6)] =0

        return X

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

            ax.axhline(y=0, color='gray')


        # Time domain signal
        x_sum, x = self.x(self.n())
        x_isum, x_i = self.x(self.ni())

        col = ['C5', 'C7']
        line = '-'

        [self.ax_time[0].plot(self.ni(), s, line, color=col[k])
         for k, s in enumerate(x_i)]

        [self.ax_time[0].plot(self.n(), s,'o', color=col[k])
         for k, s in enumerate(x)]

        self.ax_time[1].plot(self.ni(), x_isum, line, color='black')
        self.ax_time[1].plot(self.n(), x_sum, 'o', color='black')

        c, s = self.x_m(self.n())
        ci, si = self.x_m(self.ni())

        col_cs = ['b', 'r']
        line_cs = ':'
        # self.ax_time[1].plot(self.n(), c, 'o', color = col_cs[0])
        # self.ax_time[1].plot(self.n(), s, 'o', color = col_cs[1])
        for k in range(2):
            self.ax_time[k].plot(self.ni(), ci, line_cs, color = col_cs[0], label=f'cos, m={self.m}')
            self.ax_time[k].plot(self.ni(), si, line_cs, color = col_cs[1], label=f'sin, m={self.m}')

        self.ax_time[1].legend()

        X = self.dft()

        self.ax_dft[0].stem(np.abs(X) )
        self.ax_dft[1].stem(np.degrees(np.angle(X)))


        return 0

    # Simple interactive operation
    def interact(self, m=None):
        """Scale inputs and  display results.

        For interactive operation.
        Existing values are used if a parameter is omitted.

        Parameters
        ----------
        m : int, optional
            DFT component to show
        """
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
                         constrained_layout=True,
                         num='Fourier Synthesis')

        ax_time = [fig.add_subplot(2, 2, 2*k+1) for k in range(2)]
        ax_dft = [fig.add_subplot(2, 2, 2*k+2) for k in range(2)]

        ax_time[0].set_title('Signal components with cosine-and sine-waves')
        ax_time[1].set_title('Signal with cosine-and sine-waves')
        ax_dft[0].set_title('DFT magnitude')
        ax_dft[1].set_title('DFT phase')

        n_range =[-0.5, self.n_samples-0.5]
        for ax in ax_time:
            ax.set(xlabel='n',
                   xlim= n_range )

        for ax in ax_dft:
            ax.set(xlabel='m',
                   xlim= n_range )

        ax_dft[1].set(ylim=(-90, 90),
                      yticks= range(-90, 93, 30))

        ax_dft[0].set_ylabel('Magnitude')
        ax_dft[1].set_ylabel('Phase [deg]')

        return ax_time, ax_dft

    # Interactive widgets
    def _create_widgets(self):
        """Create widgets for interactive operation."""
        # Title
        title = ('DFT illustration')
        title_widget = widgets.Label(title, style=dict(font_weight='bold'))

        # Layouts definitions
        text_layout = {'continuous_update': False,
                       'layout': widgets.Layout(width='20%'),
                       'style': {'description_width': '70%'}}

        # Individual widgets
        m_widget = widgets.IntText(
            min=0, max=self.n_samples, value=1,
            description='Coefficient $m$',
            **text_layout)

        widget_layout = widgets.VBox([title_widget, m_widget])

        # Export as dictionary
        widget = {'m': m_widget}

        w = WidgetLayout(widget_layout, widget)

        return w
