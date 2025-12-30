# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 09:52:32 2025

@author: larsh
"""

import numpy as np
from math import pi, sqrt
import matplotlib.pyplot as plt
import ipywidgets


class WidgetLayout():
    """Container for widgets and layout."""

    def __init__(self, layout, widget):
        self.layout = layout
        self.widget = widget


class FrequencyResponse():
    """Demonstation of second order system frequency response.

    All calculations and plotting routines are contained in this class
    """

    def __init__(self, initialise_graphs=True, create_widgets=False):
        """Initialise system parameters."""
        self.f0 = 1         # Resonance frequency
        self.zeta = 0.7     # Damping ratio
        self.n_f = 300        # Number of points in frequency vectors
        self.flim = [-1.5, 1.5]   # Frequency limits, logarithmic

        if initialise_graphs:
            self.ax = self.initialise_graphs()

        if create_widgets:
            self.widget = self._create_widgets()

        return

    def initialise_graphs(self):
        """Initialise result graph ."""
        fig = plt.figure(figsize=[14, 7],
                         constrained_layout=True,
                         num='Second Order System - Frequency Response')
        ax = fig.subplots(2, 1, sharex=True)

        # Scale and label axes
        ax[1].set(xlabel='Frequency [Hz]')
        for a in ax:    # Common for both plots
            a.set(xlim=(np.min(self.f()),  np.max(self.f())))
            a.grid(True, which='major', axis='both')
            a.grid(True, which='minor', axis='x')

        db_min = -24
        db_max = 20
        ax[0].set(ylabel='Magnitude [dB]',
                  ylim=(db_min, db_max),
                  yticks=np.arange(db_min, db_max, 3))

        phi_min = -180
        ax[1].set(ylabel='Phase [Degrees]',
                  ylim=(phi_min, 0),
                  yticks=np.arange(phi_min, 1, 30))

        return ax

    def undercritical(self):
        """Determine whether damping is under- or overcritical."""
        return self.zeta < 1

    def f(self):
        """Create frequency vector."""
        return np.logspace(min(self.flim), max(self.flim), self.n_f)

    def H(self):
        """Calculate frequency response."""
        return 1/(1 - (self.f()/self.f0)**2 + 2j*self.zeta*self.f()/self.f0)

    def display(self):
        """Plot result in graph."""
        # Remode previous graphs and texts
        for ax in self.ax:
            for art in list(ax.lines):
                art.remove()
            for art in list(ax.texts):
                art.remove()
            for art in list(ax.patches):
                art.remove()

        # Plot magnitude and phase responses
        h_db = 20*np.log10(abs(self.H()))
        self.ax[0].semilogx(self.f(), h_db, '-', color='C0')

        self.ax[1].semilogx(self.f(),
                            np.degrees(np.angle(self.H())),
                            '-', color='C0')

        # Texts and marker lines
        indicator_color = 'gray'
        for ax in self.ax:
            ax.axvline(x=self.f0, color=indicator_color, linestyle='-')

        self.ax[0].axhline(y=0, color='gray', linestyle='-')
        y_lim = [-3, 3]
        for y in y_lim:
            self.ax[0].axhline(y=y, color=indicator_color, linestyle=':')
        self.ax[0].axhspan(y_lim[0], y_lim[1], color='green', alpha=0.1)

        self.ax[0].text(
            self.f0, -20, f' $ f_0$ = {self.f0:.3g} Hz', color='black')

        self.ax[1].axvline(x=self.f0, color=indicator_color, linestyle='-')
        self.ax[1].axhline(y=-90, color=indicator_color, linestyle='-')

        # Notify whether underdamped
        if self.undercritical():
            status = ' Undercritical '
            status_colors = ['white', 'C0']
        else:
            status = ' Overcritical '
            status_colors = ['black', 'C1']

        self.ax[0].text(5, 12, status,
                        color=status_colors[0],
                        backgroundcolor=status_colors[1])

        return

    def interact(self, f0=None, zeta=None):
        """Set values and call plotting function."""
        if f0 is not None:
            self.f0 = f0
        if zeta is not None:
            self.zeta = zeta

        self.display()

        return

    def _create_widgets(self):
        """Create widgets for interactive operation."""
        slider_layout = {'continuous_update': True,
                         'layout': ipywidgets.Layout(width='70%'),
                         'style': {'description_width': '20%'}}

        title_widget = ipywidgets.Label(
            'Second Order System Frequency Response',
            style=dict(font_weight='bold'))

        f0_widget = ipywidgets.FloatSlider(
            min=0.1, max=4, value=1, step=0.1,
            description='Resonance Frequency [Hz]',
            readout_format='.1f',
            **slider_layout)

        zeta_widget = ipywidgets.FloatLogSlider(
            min=-2, max=1, value=0.60, step=0.02,
            description='Damping Ratio',
            readout_format='.2f',
            **slider_layout)

        widget_layout = ipywidgets.VBox([title_widget, f0_widget, zeta_widget])

        # Export as dictionary
        widget = {'f0': f0_widget,
                  'zeta': zeta_widget}

        w = WidgetLayout(widget_layout, widget)

        return w
