# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 14:53:16 2024

@author: larsh
"""

import numpy as np
import matplotlib.pyplot as plt
from cmath import pi, exp, sqrt     # For readability, also covered by numpy

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
    amax : Float, optional
        Maximum scale on Re- and Im-axis

    ax : Axes object of Matplotlib, optional
        Axis to plot the results. A new axis is created if not specified

    Returns
    -------
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

    ax.grid(visible=True, which='major', axis='both')

    return ax


def plot_phasor(zk, include_sum, include_label, ax):
    """Show complex numbers as phasors in the complex plane (Argand-diagram).

    Parameters
    ----------
    zk : complex or list of complex
        Complex numbers to show

    include_sum=True : Boolean, optional
        Show the sum of the numbers in the plots

    include_label=False : Boolean, optional
        Label the phasors z1, z2, ...

    ax=None : Axes object of Matplotlib, optional
        Axis to plot the results. A new axis is created if not specified

    Returns
    -------
    ax : Axes object of Matplotlib
        Handle to axes containing the plot
    """

    # Convert input data to NumPy array
    single = np.isscalar(zk)
    if single:
        zk = np.array([zk])
        include_sum = False     # Sum is meaningless for scalar

    if isinstance(zk, list):
        zk = np.array(zk)

    # Find maximum and draw complex plane with unit circle
    amax = np.max(abs(zk))
    if include_sum:
        z_sum = np.sum(zk)
        amax = max(amax, abs(z_sum))

    ax = unitcircle(1.1*amax, ax)

    # Draw complex vactors
    k = 0
    for z in zk:
        ax.arrow(0, 0, z.real, z.imag,
                 color=COLOR_PHASORS,
                 width=PHASOR_WIDTH,
                 head_width=10*PHASOR_WIDTH,
                 length_includes_head=True)

        k += 1
        if include_label and not single:
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


def plot_signal(zk, frequency=1,
                include_sum=True,
                include_label=False,
                ax=None):
    """Show complex amplitudes and resulting signals.

    Parameters
    ----------
    zk : complex or list of complex
        Complex numbers to show

    frequency=1 : Float
        Frequency used to plot the signals

    include_sum=True : Boolean, optional
        Show the sum of the numbers in the plots

    include_label=False : Boolean, optional
        Label the phasors z1, z2, ...

    ax=None : Axes object of Matplotlib, optional
        Axis to plot the results. A new axis is created if not specified

    Returns
    -------
    ax : Axes object of Matplotlib
        Handle to axes containing the plot
    """

    t_end = 1/frequency
    fs = 32*frequency
    t = np.arange(-t_end, t_end, 1/fs)

    xc = np.zeros((len(t), len(zk)), dtype=np.complex128)
    for k in range(len(zk)):
        xc[:, k] = zk[k] * np.exp(2j*pi*frequency*t)

    x = xc.real

    ax.plot(t, x, color=COLOR_PHASORS)

    if include_sum:
        xs = x.sum(axis=1)
        ax.plot(t, xs, color=COLOR_SUM)

    if include_label:
        t_delay = -np.angle(zk)/(2*pi*frequency)
        for k in range(len(zk)):
            z_id = f"$z_{k+1}$"
            ax.text(t_delay[k], abs(zk[k]), z_id,
                    color=COLOR_PHASORS,
                    backgroundcolor='white')

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")

    ax.set_box_aspect(1)
    ax.grid(visible=True, which='major', axis='both')

    return


def phasor(zk,
           include_sum=True,
           include_label=False,
           include_signal=False,
           frequency=1,
           ax=None):
    """Show complex amplitudes and resulting signals.

    Parameters
    ----------
    zk : complex or list of complex
        Complex numbers to show

    frequency=1 : Float
        Frequency used to plot the signals

    include_sum=True : Boolean, optional
        Show the sum of the numbers in the plots

    include_label=False : Boolean, optional
        Label the phasors z1, z2, ...

    include_signal=False : Boolean, optional
        Include a plot of signals as function of time

    ax=None : Axes object of Matplotlib, optional
        Axis to plot the results. A new axis is created if not specified

    Returns
    -------
    ax_phasor : Axes object of Matplotlib
        Handle to axes containing the phasor plot

    ax_signal : Axes object of Matplotlib
        Handle to axes containing the signal traces
    """
    fig = plt.figure(figsize=(10, 5))
    if include_signal:
        ax_phasor = fig.add_subplot(1, 2, 1)
        ax_signal = fig.add_subplot(1, 2, 2)
    else:
        ax_phasor = fig.add_subplot(1, 1, 1)

    plot_phasor(zk, include_sum, include_label, ax_phasor)
    if include_signal:
        plot_signal(zk, frequency, include_sum, include_label, ax_signal)

    return
