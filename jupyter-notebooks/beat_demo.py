from math import pi
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import matplotlib.patches as patches
import matplotlib.colors as mcolors


# %%
class Beat():
    """Demonstation of beat signal with spectrum

    All calculations and plotting routines are contained in this class
    """

    def __init__(self, A=[10, 10], fc=400, fd=2):
        """Initialise signal."""
        self.A = A
        self.fc = fc
        self.fd = fd
        self.fs = 11025			# Sample rate, 1/4 of the standard 44.1 kHz
        self.duration = 1

        self.ax_time, self.ax_freq, self.ax_zoom = self.initialise_graphs()

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
        ax_time = [fig.add_subplot(n_plots, 3, 3*k+1) for k in range(n_plots)]
        ax_freq = [fig.add_subplot(n_plots, 3, 3*k+2) for k in range(n_plots)]
        ax_zoom = [fig.add_subplot(n_plots, 3, 3*k+3) for k in range(n_plots)]

        f_max = 1.2*self.fc
        f_zoom = 10*self.fd
        for k in range(n_plots):
            ax_time[k].set(xlim=(0, self.duration),
                           xlabel="Time [s]")

            ax_freq[k].set(xlim=(0, f_max),
                           xlabel="Frequency [Hz]")
            ax_freq[k].grid(True)

            ax_zoom[k].set(xlim=(self.fc-f_zoom, self.fc+f_zoom),
                           xlabel="Frequency [Hz]")
            ax_zoom[k].grid(True)

        return ax_time, ax_freq, ax_zoom

    def plot(self):
        """Plot all signals and spectra."""
        # Clear old lines
        for ax in self.ax_time + self.ax_freq + self.ax_zoom:
            for art in list(ax.lines):
                art.remove()
            for art in list(ax.collections):
                art.remove()
            for art in list(ax.patches):
                art.remove()

        # Plot time traces
        n_plots = len(self.ax_time)
        a_max = 1.2 * sum(self.A)
        p_max = 0.6 * max(self.A)**2

        for k in range(n_plots):
            self.ax_time[k].plot(self.t(), self.beat()[k],
                                 "-",
                                 color="C0")
            self.ax_time[k].set(ylim=[-a_max, a_max])

        f, pxx = self.spectrum()
        for k in range(n_plots):
            self.ax_freq[k].plot(f, pxx[k],
                                 "-",
                                 color="C0")
            self.ax_zoom[k].plot(f, pxx[k],
                                 "-",
                                 color="C0")
            self.ax_freq[k].set(ylim=[0, p_max])
            self.ax_zoom[k].set(ylim=[0, p_max])

        # Plot spectra

        return 0
