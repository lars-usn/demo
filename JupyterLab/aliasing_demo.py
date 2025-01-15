"""Illustrate aliasing in the time- and frequency domains.

Use Matplotlib backend that allows interactive operation
"""

from math import pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors


# %%
class FrequencyAliasSignal():
    """Demonstation of aliasing in the time- and frequency domains.

    All calculations and plotting routines are contained in this class
    """

    def __init__(self, f=5, phase=0, fs=20):
        """Initialise signal."""
        self.f = f                          # Original frequency
        self.phase = 0.2                    # Phase [radians]
        self.fs = fs                        # Sample rate
        self.n_alias = np.arange(-10, 10)   # Alias frequency number
        self.n_t = 1000                     # Number of point in time vectors
        self.t_max = 10/self.fs               # Length of time axis
        self.ax_time, self.ax_freq = self.initialise_graphs()
        self.color = self.initialise_colors()

    def _f_alias(self):
        """Find aliases from positive frequency."""
        return self.f + self.n_alias * self.fs

    def f_all(self):
        """Find the 20 lowest unique aliasing frequencies."""
        fp = self._f_alias()
        fn = -fp      # Add negative aliases
        return np.sort(np.append(fn, fp))

    def fa(self):
        """Principal alias frequency."""
        na = np.argmin(abs(self._f_alias()))   # Index of principal alias
        self._f_alias()[na]                    # Principal alias, signed

        return self._f_alias()[na]

    def ti(self):
        """Original time vector."""
        return np.arange(0, self.t_max, 1/(100*self.f))

    def dts(self):
        """Sample time."""
        return 1/self.fs

    def ts(self):
        """Get sampled time vector, including entire time vector."""
        return np.arange(0, self.t_max+self.dts()/2, self.dts())

    def original(self):
        """Original signal."""
        return np.cos(2 * pi * self.f * self.ti() + self.phase)

    def sampled(self):
        """Calculate sampled signal."""
        return np.cos(2*pi * self.f * self.ts() + self.phase)

    def reconstructed(self):
        """Reconstructed signal."""
        return np.cos(2 * pi * self.fa() * self.ti() + self.phase)

    def initialise_colors(self):
        """Create consistens set of colors for the plots.

        Using Matplotlib's 'Tableau' palette
        """
        color = {}
        color["original"] = mcolors.TABLEAU_COLORS['tab:orange']
        color["sampled"] = mcolors.TABLEAU_COLORS['tab:blue']
        color["reconstructed"] = mcolors.TABLEAU_COLORS['tab:green']
        color["aliased"] = mcolors.TABLEAU_COLORS['tab:red']
        color["nyquist"] = mcolors.TABLEAU_COLORS['tab:green']

        return color

    def initialise_graphs(self):
        """Initialise graphs for signals and spectra."""
        plt.close("all")
        plt.rc('font', size=10)          # Default text sizes
        fig = plt.figure(figsize=[10, 6],
                         constrained_layout=True,
                         num="Aliasing Demo")

        n_plots = 3
        ax_time = [fig.add_subplot(2, n_plots, k+1) for k in range(n_plots)]
        ax_freq = [fig.add_subplot(2, n_plots, k+1+n_plots)
                   for k in range(n_plots)]

        fs_scale = 2*self.fs
        for k in range(n_plots):
            ax_time[k].set(xlim=(0, 2/self.f),
                           ylim=(-1.1, 1.1),
                           xlabel="Time [s]")

            ax_freq[k].set(xlim=(-fs_scale, fs_scale),
                           ylim=(0, 1.1),
                           xlabel="Frequency [Hz]")

        return ax_time, ax_freq

    def plot(self):
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
        self.ax_time[0].plot(self.ti(), self.original(),
                             "-",
                             color=self.color["original"])
        self.ax_time[1].stem(self.ts(), self.sampled(),
                             linefmt=self.color["sampled"])
        self.ax_time[1].plot(self.ti(), self.original(),
                             linestyle="-",
                             color=self.color["original"])
        self.ax_time[1].plot(self.ti(), self.reconstructed(),
                             linestyle="--",
                             color=self.color["reconstructed"])
        self.ax_time[2].plot(self.ti(), self.reconstructed(),
                             linestyle="-",
                             color=self.color["reconstructed"])

        # Add titles with values
        self.ax_time[0].set_title(f"Frequency = {self.f:.1f} Hz")
        self.ax_time[1].set_title(f"Sampling at {self.fs:.1f} samples/s")
        # reconstructed_title = self.ax_time[2].set_title(
        #    f"Reconstructed frequency = {abs(self.fa()):.1f} Hz")

        # Plot spectra
        self.ax_freq[0].stem([-self.f, self.f], np.ones(2),
                             linefmt=self.color["original"])
        self.ax_freq[1].stem(self.f_all(), np.ones(len(self.f_all())),
                             linefmt=self.color["sampled"])
        self.ax_freq[1].stem([-self.f, self.f], np.ones(2),
                             linefmt=self.color["original"])
        self.ax_freq[2].stem([-self.fa(), self.fa()], np.ones(2),
                             linefmt=self.color["reconstructed"])

        # == Disabled ===
        # Indicate if aliasing occurs
        # aliasing = (self.f > self.fs/2)
        # if aliasing:
        #    nyquistcolor = self.color["aliased"]
        # else:
        #    nyquistcolor = self.color["nyquist"]
        # ===============

        # Make box showing Nyquist limits
        fn = self.fs/2
        for ax in self.ax_freq:
            ax.add_patch(patches.Rectangle((-fn, 0),       # (x,y)
                                           2*fn,           # width
                                           2,              # height
                                           alpha=0.20,     # transparency
                                           color=self.color["nyquist"]))
            ax.plot([-fn, -fn], [0, 2], color=self.color["nyquist"])
            ax.plot([fn, fn], [0, 2], color=self.color["nyquist"])

        return 0


# %%
class MultipleAliasSignal():
    """Demonstation of multiple aliasing frequencies.

    All calculations and plotting routines are contained in this class
    """

    def __init__(self, f=0.2, phase=0, fs=1):
        """Initialise signal."""
        self.f = f                          # Original frequency
        self.phase = 0.2                    # Phase [radians]
        self.fs = fs                        # Sample rate
        self.m = 0                          # Alias frequencies
        self.n_t = 1000                     # Number of point in time vectors
        self.t_max = 2/self.f               # Length of time axis
        self.ax = self.initialise_graphs()
        self.color = self.initialise_colors()

    def fa(self):
        """Find positive alias of positive frequency."""
        return (self.f + self.m * self.fs)

    def ti(self):
        """Original time vector."""
        return np.arange(0, self.t_max, 1/(300*self.f))

    def ts(self):
        """Get sampled time vector, including entire time vector."""
        return np.arange(0, self.t_max+0.5/self.fs, 1/self.fs)

    def original(self):
        """Original signal."""
        return np.cos(2 * pi * self.f * self.ti() + self.phase)

    def sampled(self):
        """Calculate sampled signal."""
        return np.cos(2*pi * self.f * self.ts() + self.phase)

    def alias(self):
        """Reconstructed signal."""
        return np.cos(2 * pi * self.fa() * self.ti() + self.phase)

    def initialise_colors(self):
        """Create consistens set of colors for the plots."""
        color = {}
        color["original"] = mcolors.TABLEAU_COLORS['tab:orange']
        color["sampled"] = mcolors.TABLEAU_COLORS['tab:blue']
        color["alias"] = mcolors.TABLEAU_COLORS['tab:green']

        return color

    def initialise_graphs(self):
        """Initialise graphs for signals and spectra."""
        plt.close("all")
        plt.rc('font', size=12)
        fig = plt.figure(figsize=[12, 6], clear=True)
        ax = fig.add_subplot(1, 1, 1)
        ax.set(xlim=(0, 2/self.f),
               ylim=(-1.1, 1.1),
               xlabel="Time [s]")

        return ax

    def plot(self):
        """Plot all signals and spectra."""
        # Clear old lines
        for art in list(self.ax.lines):
            art.remove()
        for art in list(self.ax.collections):
            art.remove()

        # Plot time traces
        self.ax.stem(self.ts(), self.sampled(),
                     linefmt=self.color["sampled"])
        self.ax.plot(self.ti(), self.original(),
                     linestyle="-",
                     color=self.color["original"],
                     label=f"f = {self.f:.2f} Hz")
        if self.m != 0:
            self.ax.plot(self.ti(), self.alias(),
                         linestyle="-",
                         color=self.color["alias"],
                         label=f"f = {abs(self.fa()):.2f} Hz")

        self.ax.legend(loc="upper right")

        return 0
