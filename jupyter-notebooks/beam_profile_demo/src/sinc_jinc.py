# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 09:12:31 2025

@author: larsh
"""

# Libraries
import numpy as np
import scipy.special as sp
from scipy.optimize import root
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def jinc(x):
    """jinc-function, Bessel-version of sinc, 2J_1(x)/x."""
    x[abs(x) < 1e-8] = 1e-8
    j = 2 * sp.jn(1, np.pi*x)/(np.pi * x)
    j[abs(x) < 1e-8] = 1.0
    return j


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

    ref = [-6, -3]
    for v in ref:
        ax.axhline(y=v, xmin=0.4, xmax=0.6, color='grey', linestyle='solid')

    ax.yaxis.set_major_locator(MultipleLocator(db_sep))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))

    ax.grid(visible=True, which='major', axis='y')

    return


def configure_axes(ax, xmax):
    """Configure axis formats."""
    ax.axhline(y=0, color='dimgray')
    ref = [0.5, 1/np.sqrt(2)]
    for v in ref:
        ax.axhline(y=v, xmin=0.4, xmax=0.6, color='grey', linestyle='solid')

    ax.set(xlim=(-xmax, xmax))
    ax.grid(True)

    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))

    ax.grid(visible=True, which='major', axis='y')

    ax.legend()

    return 0


def find_sidelobe(y, x):
    """Find first sidelobe of function y(x)."""
    k_max, _ = find_peaks(np.abs(y[x > 0]), height=0)
    x_1 = x[x > 0][min(k_max)]
    s_1 = y[x > 0][min(k_max)]

    return x_1, s_1


def plot():
    """Plot sinc and jinc functions with reference values."""
    # --- Start code to plot functions ---
    xmax = 6
    x = np.linspace(-xmax, xmax, 301)
    s = np.sinc(x)
    j = jinc(x)

    fig = plt.figure(figsize=[12, 6], constrained_layout=True)

    # Functions
    ax_lin = fig.add_subplot(3, 1, 1)
    ax_lin.plot(x, s, label='sinc(x)')
    ax_lin.plot(x, j, label='jinc(x)')
    configure_axes(ax_lin, xmax)
    ax_lin.set_ylabel('Function values')

    # MAgnitudes
    ax_abs = fig.add_subplot(3, 1, 2)
    ax_abs.plot(x, np.abs(s), label='|sinc(x)|')
    ax_abs.plot(x, np.abs(j), label='|jinc(x)|')
    configure_axes(ax_abs, xmax)
    ax_abs.set_ylabel('Magnitude')

    # dB-values
    ax_db = fig.add_subplot(3, 1, 3)
    ax_db.plot(x, db(s, p_ref=0), label='sinc(x)')
    ax_db.plot(x, db(j, p_ref=0), label='jinc(x)')
    db_axis(ax_db, db_min=-30, db_max=0, db_sep=6)
    ax_db.set(xlim=(-xmax, xmax),
              ylabel='dB re. max')
    ax_db.grid(True)
    ax_db.legend()

    # Find reference values
    val = np.array([0, 0.5, 1/np.sqrt(2)])  # Reference values
    # Corresponding dB-values as strings
    db_val = [' ', '-6 dB', '-3 dB']

    j_ref = [root(lambda x: jinc(x)-v, x0=0.5).x[0] for v in val]
    s_ref = [root(lambda x: np.sinc(x)-v, x0=0.5).x[0] for v in val]

    print(' ')
    print('Reference values ')
    print(f'{'Value':5}  {' ':6}  {'sinc':5}  {'jinc':5} ')
    for k in range(len(val)):
        print(f'{val[k]:5.3f}  {db_val[k]:6}  {
              s_ref[k]:5.3f}  {j_ref[k]:5.3f}')

    # Find first sidelobe
    x_s1, s_1 = find_sidelobe(s, x)
    x_j1, j_1 = find_sidelobe(j, x)

    s_1_db = db(s_1, p_ref=1)
    j_1_db = db(j_1, p_ref=1)

    print(' ')
    print(f'{'Sidelobes':9}  {'x':5}  {'f(x)':6} ')
    print(f'{'sinc(x)':9}  {x_s1:5.3f}  {s_1:6.3f}  {s_1_db:5.1f} dB')
    print(f'{'jinc(x)':9}  {x_j1:5.3f}  {j_1:6.3f}  {j_1_db:5.1f} dB')
