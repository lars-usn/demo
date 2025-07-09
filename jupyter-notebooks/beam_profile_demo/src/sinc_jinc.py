# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 09:12:31 2025

@author: larsh
"""

# Libraries
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import curve_analysis as ca


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


def plot():
    """Plot sinc and jinc functions with reference values."""
    # --- Start code to plot functions ---
    xmax = 6
    x = np.linspace(-xmax, xmax, 301)
    s = np.sinc(x)
    j = ca.jinc(x)

    fig = plt.figure(figsize=[12, 6], constrained_layout=True)

    # Functions
    ax_lin = fig.add_subplot(3, 1, 1)
    ax_lin.plot(x, s, label='sinc(x)')
    ax_lin.plot(x, j, label='jinc(x)')
    configure_axes(ax_lin, xmax)
    ax_lin.set_ylabel('Function values')

    # Magnitudes
    ax_abs = fig.add_subplot(3, 1, 2)
    ax_abs.plot(x, np.abs(s), label='|sinc(x)|')
    ax_abs.plot(x, np.abs(j), label='|jinc(x)|')
    configure_axes(ax_abs, xmax)
    ax_abs.set_ylabel('Magnitude')

    # dB-values
    ax_db = fig.add_subplot(3, 1, 3)
    ax_db.plot(x, ca.db(s, p_ref=0), label='sinc(x)')
    ax_db.plot(x, ca.db(j, p_ref=0), label='jinc(x)')
    ca.db_axis(ax_db, db_min=-30, db_max=0, db_sep=6)
    ax_db.set(xlim=(-xmax, xmax),
              ylabel='dB re. max')
    ax_db.grid(True)
    ax_db.legend()

    # Find reference values

    s_a = ca.Refpoints(x=x, y=s)
    j_a = ca.Refpoints(x=x, y=j)

    x_s1, s_1 = s_a.sidelobe()
    x_j1, j_1 = j_a.sidelobe()

    s_1_db = ca.db(s_1, p_ref=1)
    j_1_db = ca.db(j_1, p_ref=1)

    s_ref, y_ref = s_a.ref_values()
    j_ref, y_ref = j_a.ref_values()

    print(' ')
    print('Reference values ')
    print(f'{'Value':7}  {'sinc':7}  {'jinc':7} ')
    print(f'{y_ref:7.5f} {s_ref[0]:7.5f}  {j_ref[0]:7.5f}')
    print(f'{y_ref:7.5f} {s_ref[1]:7.5f}  {j_ref[1]:7.5f}')

    print(' ')
    print(f'{'Sidelobes':9}  {'x':5}  {'f(x)':6} ')
    print(f'{'sinc(x)':9}  {x_s1:5.3f}  {s_1:6.3f}  {s_1_db:5.1f} dB')
    print(f'{'jinc(x)':9}  {x_j1:5.3f}  {j_1:6.3f}  {j_1_db:5.1f} dB')

    return s, j, x
