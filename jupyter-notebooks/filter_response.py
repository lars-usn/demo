"""Find amd plot responses of an IIR filter."""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from math import pi


class FilterResponse():
    """Demonstration of periodic signals."""

    def __init__(self):
        self.a = [1]
        self.b = [1, 1, 1, 1]
        self.n_samples = 20
        self.n_plots = 4
        self.w = np.linspace(-pi, pi, 301)

        self.ax = self.initialise_graphs()

    def initialise_graphs(self):
        """Initialise graphs for signals and spectra."""
        # plt.close("all")
        fig = plt.figure(figsize=[10, 6],
                         constrained_layout=True,
                         num="Filter Response")

        # Define 4 subplots for responses
        ax = [fig.add_subplot(2, 2, k+1) for k in range(self.n_plots)]

        # Configure axes
        # Frequency responses
        for axis in [ax[1], ax[3]]:
            axis.set(xlabel=r"$\hat \omega / \pi$",
                     xlim=(-1, 1))

        ax[1].set(ylabel=("$|H|$"))

        ax[3].set(ylabel=r"$\angle H$ [rad/$\pi$]",
                  ylim=(-1, 1))

        # Impulse response
        ax[2].set(xlabel=r"$n$",
                  ylabel=r"$h[n]$",
                  xticks=np.arange(0, self.n_samples+1, 2))

        # Common for all plots
        for k in range(len(ax)):
            ax[k].grid(True)

        return ax

    # Private methods
    def _draw_unitcircle(self, ax):
        """Draw the unit circle."""
        circ = plt.Circle((0, 0), radius=1,
                          edgecolor="gray",
                          facecolor='None')

        # Draw circle
        ax.set_aspect(1)
        ax.add_artist(circ)
        ax.set(xlim=[-2, 2],
               ylim=[-2, 2])

        # Draw and format axes
        ax.axhline(y=0, color="gray")
        ax.axvline(x=0, color="gray")

        ax.set(xlabel="Re {z}",
               ylabel="Im {z}")

        ax.grid(visible=True, which='major', axis='both')

        return ax

    def _find_roots(self, poly, n_max):
        """Find roots and their order for polynomial."""
        poly = np.pad(poly, (0, n_max - len(poly)))
        z = np.roots(poly)
        z = np.round(z, 6).astype(complex)

        z, n_z = np.unique(z, return_counts=True)

        return z, n_z

    def _plot_pz(self, ax, z, n_z, marker, color="C0"):
        """Draw poles or zeros in specified plot."""
        if marker == 'o':
            facecolor = 'none'    # Open circles
        else:
            facecolor = color

        ax.scatter(z.real, z.imag, marker=marker, s=100,
                   color=color,
                   facecolor=facecolor)

        for k in range(len(n_z)):
            if n_z[k] > 1:
                ax.text(z[k].real, z[k].imag, f"  ({n_z[k]})  ",
                        verticalalignment="bottom",
                        horizontalalignment="left",
                        color=color)

        return 0

    # Public methods
    def n(self):
        """Array of sample numbers."""
        return np.arange(self.n_samples)

    def h(self):
        """Calculate impulse responser."""
        x = np.zeros(self.n_samples)
        x[0] = 1

        h = signal.lfilter(self.b, self.a, x)

        return h

    def H(self):
        """Calculate Frequency response from fiter coefficients."""
        # Create exponentials
        n = max(len(self.b), len(self.a))
        e = [np.exp(-1j * self.w * k) for k in range(n)]

        B = 0
        A = 0
        for k in range(len(self.b)):
            B = B + self.b[k] * e[k]

        for k in range(len(self.a)):
            A = A + self.a[k] * e[k]

        H = B/A

        H[abs(H) < 1e-10] = 0

        return H

    def pz(self):
        """Find poles and zeros."""
        n_max = max(len(self.b), len(self.a))
        z, n_z = self._find_roots(self.b, n_max)
        p, n_p = self._find_roots(self.a, n_max)

        return p, z, n_p, n_z

    def plot(self):
        """Plot all signals and spectra."""
        # Clear old lines
        for k in range(len(self.ax)):
            for art in list(self.ax[k].lines):
                art.remove()
            for art in list(self.ax[k].collections):
                art.remove()
            for art in list(self.ax[k].patches):
                art.remove()
            for art in list(self.ax[k].texts):
                art.remove()

        col = "C0"

        # Pole-zero plot
        self._draw_unitcircle(self.ax[0])
        p, z, n_p, n_z = self.pz()

        self._plot_pz(self.ax[0], p, n_p, "x")
        self._plot_pz(self.ax[0], z, n_z, "o")

        p_max = np.max(np.abs(p))
        z_max = np.max(np.abs(z))

        pz_scale = 1.5*max(p_max, z_max, 1)
        self.ax[0].set(xlim=(-pz_scale, pz_scale),
                       ylim=(-pz_scale, pz_scale))

        stable = (p_max < 1)
        if stable:
            stabletext = "Stable"
            stablecolor = "green"
        else:
            stabletext = "Unstable"
            stablecolor = "red"

        self.ax[0].set_title(stabletext, color=stablecolor)

        # Impuklse response
        h_max = 1.2*np.max(abs(self.h()))
        self.ax[2].stem(self.n(), self.h(), markerfmt=col)
        self.ax[2].set(xlim=(-0.5, self.n_samples),
                       ylim=(-h_max, h_max))

        # Frequency response
        if stable:
            H_mag = np.abs(self.H())
            self.ax[1].plot(self.w/pi, H_mag, color=col)
            self.ax[1].set_ylim(0, np.max(H_mag))

            self.ax[3].plot(self.w/pi, np.angle(self.H())/pi, color=col)

        # Marker lines
        for k in [1, 2, 3]:
            self.ax[k].axvline(x=0, color="gray")
            self.ax[k].axhline(y=0, color="gray")

        return 0
