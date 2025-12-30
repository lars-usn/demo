from math import pi
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import ipywidgets


class WidgetLayout():
    """Container for widgets and layout."""

    def __init__(self, layout, widget):
        self.layout = layout
        self.widget = widget


class Beat():
    """Demonstation of aliasing in the time- and frequency domains.

    All calculations and plotting routines are contained in this class
    """

    def __init__(self, initialise_graphs=True, create_widgets=False):
        """Initialise system parameters."""
        self.a = [1.0, 1.0]
        self.f = [400, 410]
        self.fs = 11025			# Sample rate, 1/4 of the standard 44.1 kHz
        self.duration = 1

        if initialise_graphs:
            self.ax_time, self.ax_freq = self.initialise_graphs()

        if create_widgets:
            self.widget = self._create_widgets()

        return

    def dt(self):
        """Sample time."""
        return 1/self.fs

    def t(self):
        """Create time vector."""
        return np.arange(0, self.duration, self.dt())

    def beat(self):
        """Synthesize a beat tone from two cosine waves."""
        s = [self.a[k] * np.cos(2*pi * self.f[k] * self.t())
             for k in range(2)]

        s.append(s[0]+s[1])

        return s

    def spectrum(self):
        """Calculate power spectrum of signals."""
        f, pxx = signal.periodogram(self.beat(), self.fs)
        return f, pxx

    def initialise_graphs(self):
        """Initialise graphs for signals and spectra."""
        plt.close("all")
        plt.rc('font', size=10)          # Default text sizes
        fig = plt.figure(figsize=[14, 6],
                         constrained_layout=True,
                         num="Beat Demo")

        n_plots = 3
        ax_time = [fig.add_subplot(n_plots, 2, 2*k+1) for k in range(n_plots)]
        ax_freq = [fig.add_subplot(n_plots, 2, 2*k+2) for k in range(n_plots)]

        for a in ax_time:
            a.sharex(ax_time[0])
            a.set(xlim=(0, self.duration/4))

        for a in ax_freq:
            a.sharex(ax_freq[0])
            a.set(xlim=(0, 1.4*self.f[0]),
                  ylim=(0, 0.7))
            a.grid(True, which='major', axis='both')
            a.grid(True, which='minor', axis='x')

        ax_time[2].set(xlabel="Time [s]")
        ax_freq[2].set(xlabel="Frequency [Hz]")

        return ax_time, ax_freq

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

        # Plot time traces
        n_plots = len(self.ax_time)
        for k in range(n_plots):
            self.ax_time[k].plot(self.t(), self.beat()[k],
                                 "-",
                                 color="C0")

        for k in range(n_plots-1):
            self.ax_time[k].set(
                title=fr'Signal {k+1},   $s_{k+1}(t)$,     $f_{k+1}$={self.f[k]:.0f} Hz')

        self.ax_time[2].set(
            title=fr'Summed signal, $s_{1}(t) + s_{2}(t)$')

        f, pxx = self.spectrum()
        for k in range(n_plots):
            self.ax_freq[k].plot(f, pxx[k],
                                 "-",
                                 color="C0")

        return 0

    def interact(self, f1=None, f2=None):
        """Set values and call plotting function."""
        if f1 is not None:
            self.f[0] = f1
        if f2 is not None:
            self.f[1] = f2

        self.display()

        return

    def _create_widgets(self):
        """Create widgets for interactive operation."""
        text_layout = {'continuous_update': True,
                       'layout': ipywidgets.Layout(width='15%'),
                       'style': {'description_width': '60%'}}

        title_widget = ipywidgets.Label('Beat',
                                        style=dict(font_weight='bold'))

        f_widget = [ipywidgets.FloatText(min=300, max=500,
                                         value=self.f[k], step=1.0,
                                         description=f'Frequency {k+1} [Hz]',
                                         readout_format='.0f',
                                         **text_layout)
                    for k in range(len(self.f))]

        widget_layout = ipywidgets.VBox([title_widget] + f_widget)

        widget = {'f1': f_widget[0],
                  'f2': f_widget[1]}

        w = WidgetLayout(widget_layout, widget)

        return w
