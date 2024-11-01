# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 14:53:16 2024

@author: larsh
"""

import numpy as np
import matplotlib.pyplot as plt
from cmath import pi, exp, sqrt     # For readability, also covered by numpy

# Colors and line widths
COLOR_UNIT_CIRCLE = 'gray'
COLOR_AXES = 'black'
COLOR_PHASORS = 'tab:blue'
COLOR_SUM = 'tab:orange'

AXIS_WIDTH = 0.01
PHASOR_WIDTH = 0.02


def unitcircle(amax=1.5, ax=None):
    """Draw the unit circle.

    Parameters
    ----------
    amax : Float, optional
        Maximum scale on Re- and Im-axis

    ax : Matplotlib axes, optional
        Axis to plot the results. A new axis is created if not specified

    Returns
    -------
    ax : Matplotlib axes
        Handle to axes containing the plot
    """
    # Create new figure
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    circ = plt.Circle((0, 0), radius=1,
                      edgecolor=COLOR_UNIT_CIRCLE,
                      facecolor='None')

    # Draw circle
    ax.set_aspect(1)
    ax.add_artist(circ)

    # Draw axes
    # ax.plot([0, 0], [-amax, amax], color=COLOR_AXES, linewidth=AXIS_WIDTH)
    # ax.plot([-amax, amax], [0, 0], color=COLOR_AXES, linewidth=AXIS_WIDTH)

    # Draw axes as arrows
    ax.arrow(-amax, 0, 2*amax, 0,
             color=COLOR_AXES,
             width=AXIS_WIDTH,
             head_width=10*AXIS_WIDTH,
             length_includes_head=True)

    ax.arrow(0, -amax, 0, 2*amax,
             color=COLOR_AXES,
             width=AXIS_WIDTH,
             head_width=10*AXIS_WIDTH,
             length_includes_head=True)

    ax.text(0.97*amax, -0.03*amax, " Re{z} ",
            verticalalignment='top',
            horizontalalignment='right',
            color=COLOR_AXES)

    ax.text(0, 0.95*amax, " Im{z} ",
            verticalalignment='top',
            horizontalalignment='right',
            color=COLOR_AXES)

    # Format axes
    ax.set_xlim(-amax, amax)
    ax.set_ylim(-amax, amax)

    # ax.set_xlabel("Re {z}")
    # ax.set_ylabel("Im {z}")

    ax.grid(visible=True, which='major', axis='both')

    return ax


def plot_phasor(zk, labels, include_sum, ax):
    """Show complex numbers as phasors in the complex plane (Argand-diagram).

    Parameters
    ----------
    zk : complex or list of complex
        Complex numbers to show

    labels=[] : List of strings, optional
        List of labels to mark phasors

    include_sum=False : Boolean, optional
        Show the sum of all numbers as a phasor

    ax=None : Matplotlib axes, optional
        Axis to plot the results. A new axis is created if not specified

    Returns
    -------
    ax : Matplotlib axes
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
    for z in zk:
        ax.arrow(0, 0, z.real, z.imag,
                 color=COLOR_PHASORS,
                 width=PHASOR_WIDTH,
                 head_width=10*PHASOR_WIDTH,
                 length_includes_head=True)

    k = 0
    for label in labels:
        ax.text(zk[k].real, zk[k].imag, label, color=COLOR_PHASORS)
        k += 1

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


def plot_signal(zk, labels=[], include_sum=False, frequency=1, ax=None):
    """Show complex amplitudes and resulting signals.

    Parameters
    ----------
    zk : complex or list of complex
        Complex numbers to show

    labels=[] : List of strings, optional
        List of labels to mark phasors

    include_sum=False : Boolean, optional
        Show the sum of all numbers as a phasor

    frequency=1 : Float, optional
        Frequency used to plot the signals

    ax=None : Matplotlib axes, optional
        Axis to plot the results. A new axis is created if not specified

    Returns
    -------
    ax : Matplotlib axes
        Handle to axes containing the plots
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

    k = 0
    for label in labels:
        t_delay = -np.angle(zk[k])/(2*pi*frequency)
        ax.text(t_delay, abs(zk[k]), label,
                color=COLOR_PHASORS,
                backgroundcolor='white')
        k += 1

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")

    ax.set_box_aspect(1)
    ax.grid(visible=True, which='major', axis='both')

    return ax


def phasor(zk,
           labels=[],
           include_sum=False,
           include_signal=False,
           frequency=1):
    """Show complex amplitudes and resulting signals.

    Parameters
    ----------
    zk : complex or list of complex
        Complex numbers to show

    labels=[] : List of strings, optional
        List of labels to mark phasors

    include_sum=False : Boolean, optional
        Show the sum of all numbers as a phasor

    include_signal=False : Boolean, optional
        Include a plot of signals as function of time

    frequency=1 : Float, optional
        Frequency used to plot the signals

    Returns
    -------
    ax : List of Matplotlib Axes
        Handle to axes containing the plots
    """
    fig = plt.figure(figsize=(10, 5))
    if include_signal:
        ax_phasor = fig.add_subplot(1, 2, 1)
        ax_signal = fig.add_subplot(1, 2, 2)
        ax = [ax_phasor, ax_signal]
    else:
        ax_phasor = fig.add_subplot(1, 1, 1)
        ax = [ax_phasor]

    plot_phasor(zk, labels, include_sum, ax_phasor)
    if include_signal:
        plot_signal(zk, labels, include_sum, frequency, ax_signal)

    return ax
