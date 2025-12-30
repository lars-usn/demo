from math import pi
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import matplotlib.patches as patches
import matplotlib.colors as mcolors


class Beat():
    """Demonstation of aliasing in the time- and frequency domains.

    All calculations and plotting routines are contained in this class
    """

    def __init__(self):
        """Initialise signal."""
        self.A = [10, 10]
        self.fc = 100
        self.fd = 4
        self.fs = 11025			# Sample rate, 1/4 of the standard 44.1 kHz
        self.duration = 1

        self.ax_time, self.ax_freq = self.initialise_graphs()

    def dt(self):
        """Sample time."""
        return 1/self.fs

    def t(self):
        """Create time vector."""
        return np.arange(0, self.duration, self.dt())

    def fk(self):
        """Calculate the two signal frequencies."""
        return self.fc + self.fd*np.array([-1, 1])

    def beat(self):
        """Synthesize a beat tone from two cosine waves."""
        s = [self.A[k] * np.cos(2*pi * self.fk()[k] * self.t())
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
        fig = plt.figure(figsize=[10, 6],
                         constrained_layout=True,
                         num="Aliasing Demo")

        n_plots = 3
        ax_time = [fig.add_subplot(n_plots, 2, 2*k+1) for k in range(n_plots)]
        ax_freq = [fig.add_subplot(n_plots, 2, 2*k+2) for k in range(n_plots)]

        f_max = 1.4*self.fc
        for k in range(n_plots):
            ax_time[k].set(xlim=(0, self.duration),
                           xlabel="Time [s]")

            ax_freq[k].set(xlim=(0, f_max),
                           xlabel="Frequency [Hz]")

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

        f, pxx = self.spectrum()
        for k in range(n_plots):
            self.ax_freq[k].plot(f, pxx[k],
                                 "-",
                                 color="C0")

        return 0
