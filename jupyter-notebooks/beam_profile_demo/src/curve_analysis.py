"""
Utility functions to analyse curves for ultrasound fields.

Created on Wed Jul  9 13:11:42 2025

@author: larsh
"""

import numpy as np
import scipy.special as sp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.image as mpimg


# Mathematical functions
def jinc(x):
    """jinc-function, Bessel-version of sinc, 2J_1(x)/x."""
    x[abs(x) < 1e-8] = 1e-8
    j = 2 * sp.jn(1, np.pi*x)/(np.pi * x)
    j[abs(x) < 1e-8] = 1.0
    return j


# Decibel scale
def db(p, p_ref=1e-6):
    """Decibel from pressure."""
    if p_ref == 0:
        p_ref = np.max(p)

    return 20 * np.log10(abs(p/p_ref))


def db_axis(ax, db_min=-40, db_max=0, db_sep=6):
    """Configure dB-scaled axis.

    Parameters
    ----------
    ax: axis object
        Axis to configure
    db_min: float
        Minimum on dB-axis
    db_max: float
        Maximum on dB-axis
    db_sep: float
        Separation between major ticks
    """
    ax.set_ylim(db_min, db_max)

    ax.yaxis.set_major_locator(MultipleLocator(db_sep))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))

    ax.grid(visible=True, which='major', axis='y')

    return


def db_colorbar(cbar, db_sep=6):
    """Configure dB-scaled colorbar.

    Parameters
    ----------
    cbar: Colorbar object
        Colorbar to configure
    db_sep: float
        Separation between major ticks
    """
    tick_min = (cbar.vmin // db_sep) * db_sep
    db_ticks = np.arange(tick_min, cbar.vmax+db_sep, db_sep)
    cbar.set_ticks(db_ticks)
    cbar.minorticks_off()

    return


# Curve analysis
def parabolic_max(y, x, kmax=None, check=False):
    """Subsample-interpolation by parabolic interpolation to find max.

    Parameters
    ----------
    y: array of float
        Function values
    x: array of float
        Function arguments
    k: int, optional
        Index to search for max around
    check: boolean, optional
        Plot graph to check result

    Returns
    -------
    ymax: float
        Max. value for y(x)
    xmax: float
        Argument for max y(x)
    """
    # Find max and interpolate
    if kmax is None:
        kmax = np.argmax(abs(y))

    kz = np.arange(kmax-1, kmax+2)  # 3 points around max
    yz = y[kz]
    xz = x[kz]

    # Fit parabola
    p = np.polyfit(xz, yz, 2)
    xmax = -p[1]/(2*p[0])      # Max. from differentiation pf polynomial
    ymax = np.polyval(p, xmax)

    # Plot result for inspection
    if check:
        n_pts = 1000
        xn = np.linspace(min(xz), max(xz), n_pts)
        yn = np.polyval(p, xn)
        plt.plot(xn, yn, color='C0', linestyle='--')
        plt.plot(xz, yz, color='C1', marker='x')
        plt.plot(xmax, ymax, color='red', marker='o')
        plt.grid(True)

    return xmax, ymax


class Refpoints():
    """Find reference points for y(x).

    Main lobe, -3dB and -6dB limits, zero-crossings, and largest side lobe
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def k_max(self):
        """Find index of main peak."""
        return self.y.argmax()

    def k_zero(self):
        """Find indices of zero crossings."""
        return np.where(np.diff(np.sign(self.y)))[0]

    def main_peak(self):
        """Find value and position of main lobe."""
        return parabolic_max(self.y, self.x)

    def sidelobe(self):
        """Find value and position of highest side lobe."""
        kp, _ = find_peaks(np.abs(self.y), height=0)
        kp = np.setdiff1d(kp, self.k_max())        # Remove main peak

        if np.size(kp) == 0:
            x = y = np.nan
        else:
            ks = kp[np.argmax(np.abs(self.y[kp]))]
            x, y = parabolic_max(self.y, self.x, kmax=ks)

        return x, y

    def ref_values(self, y_rel=0.5):
        """Find positions of reference values.

        Arguments
        ---------
        y_rel: float
            Reference value relative max

        Returns
        -------
        x: array of float
            Positions of reference values
        y_lim: float
            Reference value
        """
        k_max = self.k_max()
        x_max, y_max = self.main_peak()
        y_lim = abs(y_max * y_rel)

        # Start at peak, look up and down until limit is passed
        k = k_max
        while self.y[k] > y_lim and k < len(self.y)-1:
            k += 1
        k_hi = k

        k = k_max
        while self.y[k] > y_lim and k > 0:
            k -= 1
        k_lo = k

        # Seach downwards and upwards from main peak
        ni = []    # interp requires increasing argument
        ni.append(np.arange(k_lo, k_max-1, 1))
        ni.append(np.arange(k_hi, k_max, -1))

        xm = [self.x[ni[k]] for k in range(2)]
        ym = [self.y[ni[k]] for k in range(2)]

        # -6dB, -3dB and zero-crossings

        x = np.zeros(2)

        for k in range(2):
            x[k] = np.interp(y_lim, ym[k], xm[k])

        return x, y_lim

    def lobe_width(self, y_rel=0.5):
        """Find width between reference values.

        Arguments
        ---------
        y_rel: float
            Reference value relative max

        Returns
        -------
        dx: float
            Width between reference values
        y_lim: float
            Reference value
        """
        x_lim, y_lim = self.ref_values(y_rel)
        dx = x_lim[1] - x_lim[0]

        return dx, y_lim


# Figure formatting
def remove_artists(ax):
    """Clear axis of all old artists."""
    for art in list(ax.lines):
        art.remove()
    for art in list(ax.collections):
        art.remove()
    for art in list(ax.patches):
        art.remove()
    for art in list(ax.texts):
        art.remove()

    return


def remove_fig_text(fig):
    """Remove all existing text from figure."""
    for art in list(fig.texts):
        art.remove()

    return 0


def set_fig_text(fig, text, xpos=0.0, ypos=0.0,
                 background_color='whitesmoke'):
    """Remove all existing text and add new text box.

    Parameters
    ----------
    fig: figure
        Figure to put text into
    text: string
        Text to write, multiline
    xpos: float, optional
        x-position of text, rel. figure
    ypos: float, optional
        y-position of text, rel. figure
    background_color: color string, optional
        Bckground color of text-box
    """
    fig.text(xpos, ypos, text,
             fontsize='medium',
             bbox={'facecolor': background_color,
                   'boxstyle': 'Round',
                   'pad': 1})

    return 0


def add_logo(fig,
             logofile='usn-logo-purple.png',
             logopos=[0.02, 0.02, 0.2, 0.2]):
    """Add logo image to result figure.

    Parameters
    ----------
    fig: figure
        Figure to put text into
    logofile: string
        Name of image file containing logo
    logopos: 4 element list of float
        Position of logo in figure
    """
    try:
        img = mpimg.imread(logofile)
        ax_logo = fig.add_axes(logopos, anchor='SW')
        ax_logo.imshow(img)
        ax_logo.axis('off')
    except Exception:
        pass

    return
