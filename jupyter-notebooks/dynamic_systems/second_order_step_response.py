"""
Created on Tue Dec 30 09:33:04 2025

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


class StepResponse:
    """Demonstation of second order system step response.

    All calculations and plotting routines are contained in this class
    """

    def __init__(self, initialise_graphs=True, create_widgets=False):
        """Initialise system parameters."""
        self.f0 = 1         # Resonance frequency
        self.zeta = 0.7     # Damping ratio
        self.t_min = -0.2    # Minimum time on plot
        self.t_max = 4       # Maximum time on plot
        self.n_t = 300       # Number of points in time vectors

        if initialise_graphs:
            self.ax = self.initialise_graphs()

        if create_widgets:
            self.widget = self._create_widgets()

        return

    def initialise_graphs(self):
        """Initialise result graph ."""
        plt.close('all')
        # plt.rc('font', size=10)          # Default text sizes
        fig = plt.figure(figsize=[14, 5],
                         constrained_layout=True,
                         num='Second Order System - Step Response')
        ax = fig.add_subplot(1, 1, 1)

        ax.set(xlim=(self.t_min, self.t_max),
               ylim=(0, 2),
               xlabel='Time $t$ [s]',
               ylabel='Response $s(t)')

        ax.grid(True)

        return ax

    def undercritical(self):
        """Determine whether damping is under- or overcritical."""
        if self.zeta == 1:
            self.zeta += 0.001  # Make overcritical by adding tiny amount

        return self.zeta < 1

    def fd(self):
        """Calculate frequency of free oscillations."""
        if self.undercritical():
            fd = self.f0 * sqrt(1-self.zeta**2)
        else:
            fd = 0
        return fd

    def T0(self):
        """Calculate resonance period."""
        return 1/self.f0

    def Td(self):
        """Calculate period of free oscillations."""
        if self.undercritical():
            Td = 1/self.fd()
        else:
            Td = 0
        return Td

    def t(self):
        """Create time vector."""
        return np.linspace(self.t_min, self.t_max, self.n_t)

    def _calc_response(self, t):
        """Calculate step response."""
        w0 = 2*pi*self.f0

        if self.undercritical():
            zt = sqrt(1-self.zeta**2)
            wd = w0*zt        # Frequency of damped oscillations
            phi = np.arctan2(-self.zeta, zt)
            s = 1 - np.exp(-self.zeta*w0*t) * 1/zt * np.cos(wd*t + phi)
        else:
            z = sqrt(self.zeta**2-1)
            s = 1 - (self.zeta+z)/(2*z) * np.exp((-self.zeta+z)*w0*t) \
                + (self.zeta-z)/(2*z) * np.exp((-self.zeta-z)*w0*t)

        k_neg = np.argwhere(self.t() < 0)
        s[k_neg] = np.zeros_like(k_neg)   # Set initial value for t<0

        return s

    def s(self):
        """Find step response for all time-points."""
        s = self._calc_response(self.t())
        return s

    def display(self):
        """Plot result in graph."""
        # Remove existing graphs and texts
        for art in list(self.ax.lines):
            art.remove()
        for art in list(self.ax.texts):
            art.remove()

        # Plot graph
        self.ax.plot(self.t(), self.s(), linestyle='-', color='C0')
        self.ax.axhline(y=1, linestyle='-', color='black')

        # Mark resonance period
        self.ax.axvline(x=self.T0(), linestyle='--', color='C2')
        self.ax.text(self.T0(), 0.3, f' $T_0$ = {
                     self.T0():0.2f} s', color='C2')

        # Mark period of free oscillations and whether underdamped
        if self.undercritical():
            self.ax.axvline(x=self.Td(), linestyle='--', color='C3')
            self.ax.text(self.Td(), 0.1, f' $T_d$ = {
                         self.Td():0.2f} s', color='C3')
            status = ' Undercritical '
            status_colors = ['white', 'C0']
        else:
            status = ' Overcritical '
            status_colors = ['black', 'C1']

        self.ax.text(2.7, 1.75, status,
                     color=status_colors[0], backgroundcolor=status_colors[1])

        return 0

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
            'Second Order System Step Response',
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
