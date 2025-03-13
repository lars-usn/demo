"""Find amd plot responses of an IIR filter."""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from math import pi


class PoleZero:
    """Container for poles and zeros, values and order."""

    def __init__(self):
        self.value = []
        self.order = []


class FilterResponse:
    """Demonstration of IIR filter response."""

    def __init__(self):
        self.b = [1, 1, 1, 1]  # Forward coefficients
        self.a = [1]           # Reverse coefficients
        self.n_samples = 20    # No. of samples in impulse response
        self.n_plots = 4       # No.of plots in figure
        self.w = np.linspace(-pi, pi, 301)    # Frequency vector
        self.ax, self.fig = self._initialise_graphs()

    ###################################################################
    # Public methods
    def n(self):
        """Generate array of sample numbers."""
        return np.arange(self.n_samples)

    def h(self):
        """Calculate impulse responser."""
        x = signal.unit_impulse(self.n_samples)
        h = signal.lfilter(self.b, self.a, x)

        return h

    def H(self):
        """Calculate Frequency response from fiter coefficients."""
        # Create exponentials
        n = max(len(self.b), len(self.a))
        e = [np.exp(-1j * self.w * k) for k in range(n)]

        # Calculate nominator B and denominator A by summing exponentials
        B = A = 0
        for k in range(len(self.b)):
            B = B + self.b[k] * e[k]

        for k in range(len(self.a)):
            A = A + self.a[k] * e[k]

        H = B/A
        H[abs(H) < 1e-10] = 0   # Suppress roundoff-errors

        return H

    def pz(self):
        """Find poles and zeros."""
        n_max = max(len(self.b), len(self.a))
        z = self._find_roots(self.b, n_max)
        p = self._find_roots(self.a, n_max)

        return p, z

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

        col = 'C0'    # Standard color for all plots

        # Pole-zero plot
        self._draw_unitcircle(self.ax[2])
        p, z = self.pz()

        self._plot_pz(self.ax[2], p, 'poles')
        self._plot_pz(self.ax[2], z, 'zeros')

        p_max = np.max(np.abs(p.value))
        z_max = np.max(np.abs(z.value))
        pz_scale = 1.5*max(p_max, z_max, 1)
        self.ax[0].set(xlim=(-pz_scale, pz_scale),
                       ylim=(-pz_scale, pz_scale))

        # Message Stable - Unstable
        stable = (p_max < 1)
        if stable:
            stabletext = 'Stable'
            stablecolor = 'green'
        else:
            stabletext = 'Unstable'
            stablecolor = 'red'

        self.ax[2].set_title(stabletext, color=stablecolor)

        # Impuklse response
        h_max = 1.2*np.max(abs(self.h()))
        self.ax[0].stem(self.n(), self.h(), markerfmt=col)
        self.ax[0].set(xlim=(-1, self.n_samples),
                       ylim=(-h_max, h_max))

        # Frequency response, for stable system only
        if stable:
            H_mag = np.abs(self.H())
            self.ax[1].plot(self.w/pi, H_mag, color=col)
            self.ax[1].set_ylim(0, np.max(H_mag))

            self.ax[3].plot(self.w/pi, np.angle(self.H())/pi, color=col)

        # Marker lines
        for k in [0, 1, 3]:
            self.ax[k].axvline(x=0, color='gray')
            self.ax[k].axhline(y=0, color='gray')
        return

    def print(self, fname='filterresponse', w=10, h=7):
        """Print figure to file."""
        self.fig.set_size_inches(w, h)
        self.fig.savefig(fname+'.png', format="png", bbox_inches="tight")

        return

    ###################################################################
    # Non-public methods
    def _initialise_graphs(self):
        """Initialise graphs for signals and spectra."""
        # plt.close('all')
        fig = plt.figure(figsize=[10, 7],
                         constrained_layout=True,
                         num='Filter Response')

        # Define subplots for responses
        ax = [fig.add_subplot(2, 2, k+1) for k in range(self.n_plots)]

        # Configure axes
        # Impulse response
        ax[0].set(xlabel=r'$n$',
                  ylabel=r'$h[n]$',
                  xticks=np.arange(0, self.n_samples+1, 2))

        # Frequency responses
        for axis in [ax[1], ax[3]]:
            axis.set(xlabel=r'$\hat \omega / \pi$',
                     xlim=(-1, 1))

        ax[1].set(ylabel=('$|H|$'))

        ax[3].set(ylabel=r'$\angle H$ [rad/$\pi$]',
                  ylim=(-1, 1))

        # Pole-zero plot
        z_max = 1.5
        ax[2].set(aspect='equal',
                  xlim=[-z_max, z_max],
                  ylim=[-z_max, z_max],
                  xlabel='Re {$z$}',
                  ylabel='Im {$z$}')

        self._draw_unitcircle(ax[2])

        # Common for all plots
        for k in range(len(ax)):
            ax[k].grid(visible=True, which='major', axis='both')

        return ax, fig

    def _draw_unitcircle(self, ax):
        """Draw the unit circle for pole-zero plot."""
        circ = plt.Circle((0, 0), radius=1,
                          edgecolor='gray',
                          facecolor='None')

        ax.add_artist(circ)

        ax.axhline(y=0, color='gray')
        ax.axvline(x=0, color='gray')

        return ax

    def _find_roots(self, poly, n_max):
        """Find roots with order for polynomial."""
        z = PoleZero()

        poly = np.pad(poly, (0, n_max - len(poly)))
        r = np.roots(poly)
        r = np.round(r, 6).astype(complex)   # Suppress roundoff-errors

        z.value, z.order = np.unique(r, return_counts=True)

        return z

    def _plot_pz(self, ax, z, pz_type, color='C0'):
        """Draw poles or zeros in specified axis."""
        if pz_type.lower().startswith('pol'):
            marker = 'x'
            facecolor = color
        else:
            marker = 'o'
            facecolor = 'none'    # Open circles for zeros

        ax.scatter(z.value.real, z.value.imag, marker=marker, s=100,
                   color=color,
                   facecolor=facecolor)

        # Write pole/zero order if larger than 1
        for k in range(len(z.order)):
            if z.order[k] > 1:
                ax.text(z.value[k].real, z.value[k].imag, f'  ({z.order[k]})',
                        verticalalignment='bottom',
                        horizontalalignment='left',
                        color=color)

        return 0
