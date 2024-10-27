# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 14:53:16 2024

@author: larsh
"""

import numpy as np
import matplotlib.pyplot as plt
from cmath import pi, exp     # For readability, also covered by numpy

# Colors and line widths
COLOR_UNIT_CIRCLE = 'lightsteelblue'
COLOR_AXES = 'steelblue'
COLOR_PHASORS = 'orangered'
COLOR_SUM = 'maroon'

AXIS_WIDTH = 0.5
PHASOR_WIDTH = 0.01


def unitcircle(amax=1.5, ax=None):
    """Draw the unit circle

    Parameters
    ----------
    amax : Float
        Maximum scale on Re- and Im-axis

    Returns
    ax : Axes object of Matplotlib
        Handle to axes containing the plot
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    circ = plt.Circle((0, 0), radius=1,
                      edgecolor=COLOR_UNIT_CIRCLE,
                      facecolor='None')
    ax.set_aspect(1)
    ax.add_artist(circ)
    ax.plot([0, 0], [-amax, amax], color=COLOR_AXES, linewidth=AXIS_WIDTH)
    ax.plot([-amax, amax], [0, 0], color=COLOR_AXES, linewidth=AXIS_WIDTH)

    ax.set_xlim(-amax, amax)
    ax.set_ylim(-amax, amax)

    ax.set_xlabel("Re {z}")
    ax.set_ylabel("Im {z}")

    return ax


def phasor(zk, include_sum=False, ax=None):
    """Show complex numbers as phasors in the complex plane (Argand-diagram).

    Parameters
    ----------
    zk : 1D array of float
        Complex numbers to show

    include_sum : Boolean, optional
        Include the sum of the numbers

    Returns
    ax : Axes object of Matplotlib
        Handle to axes containing the plot
    """

    # Find maximum and draw complex plane with unit circle
    amax = max(abs(zk))

    if include_sum:
        z_sum = sum(zk)
        amax = 1.1*max(amax, abs(z_sum))

    ax = unitcircle(amax, ax)

    # Draw complex vactors
    k = 0
    for z in zk:
        ax.arrow(0, 0, z.real, z.imag,
                 color=COLOR_PHASORS,
                 width=PHASOR_WIDTH,
                 head_width=10*PHASOR_WIDTH,
                 length_includes_head=True)

        k += 1
        z_id = f" $z_{k}$ "
        ax.text(z.real, z.imag, z_id,
                color=COLOR_PHASORS)

    # Draw sum
    if include_sum:
        z_id = "$ \\Sigma z $"
        ax.arrow(0, 0, z_sum.real, z_sum.imag,
                 color=COLOR_SUM,
                 width=PHASOR_WIDTH,
                 head_width=10*PHASOR_WIDTH,
                 length_includes_head=True)

        ax.text(z_sum.real, z_sum.imag, z_id,
                color=COLOR_SUM)

    return ax


def signal(zk, frequency=1, include_sum=False, ax=None):
    """Show complex amplitudes and resulting signals.

    Parameters
    ----------
    zk : 1D array of float
        Complex numbers to show

    include_sum : Boolean, optional
        Include the sum of the numbers

    Returns
    ax : Axes object of Matplotlib
        Handle to axes containing the plot
    """
    fig = plt.figure()
    ax_phasor = fig.add_subplot(1, 3, 1)

    ax_phasor = phasor(zk, include_sum, ax_phasor)

    t_end = 2/frequency
    fs = 32*frequency
    t = np.arange(0, t_end, 1/fs)

    xc = np.zeros((len(t), len(zk)), dtype=np.complex128)
    for k in range(len(zk)):
        xc[:, k] = zk[k] * np.exp(2j*pi*frequency*t)

    x = xc.real

    ax_signal = fig.add_subplot(1, 3, (2, 3))
    ax_signal.plot(t, x, color=COLOR_PHASORS)

    if include_sum:
        xs = x.sum(axis=1)
        ax_signal.plot(t, xs, color=COLOR_SUM)

    ax_signal.set_xlabel("Time [s]")
    ax_signal.set_ylabel("Amplitude")

    ax_signal.grid(visible=True, which='major', axis='both')

    return
