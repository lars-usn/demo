"""
Created on Mon Dec 29 23:45:30 2025

@author: larsh
"""

import numpy as np
from math import pi
import matplotlib.pyplot as plt
import ipywidgets


class WidgetLayout():
    """Container for widgets and layout."""

    def __init__(self, layout, widget):
        self.layout = layout
        self.widget = widget


class StepResponse:
    """Demonstration of first order system step response.

    All calculations and plotting routines are contained in this class
    """

    def __init__(self, initialise_graphs=True, create_widgets=False):
        """Initialise system parameters."""
        self.s_0 = 0        # Initial value
        self.s_inf = 1    # Final value
        self.tau = 1        # Time constant
        self.t_min = -0.7       # Minimum time on plot
        self.t_max = 30       # Maximum time on plot
        self.n_t = 300        # Number of points in time vectors

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
                         num='First Order System - Step Response')
        ax = fig.add_subplot(1, 1, 1)

        ax.set(xlim=(self.t_min, self.t_max),
               xlabel='Time [s]',
               ylabel='Response s(t)')

        return ax

    def t(self):
        """Create time vector."""
        return np.linspace(self.t_min, self.t_max, self.n_t)

    def _calc_response(self, t):
        """Calculate step response."""
        if self.tau == 0:
            # Exception if time constant equals zero
            s = self.s_inf * np.ones_like(t)
        else:
            s = self.s_inf - (self.s_inf - self.s_0) * np.exp(-t/self.tau)

        k_neg = np.argwhere(t < 0)
        s[k_neg] = self.s_0 * np.ones_like(k_neg)   # Set initial value for t<0

        return s

    def s(self):
        """Find step response for all time-points."""
        s = self._calc_response(self.t())
        return s

    def display(self):
        """Plot result in graph."""
        for art in list(self.ax.lines):
            art.remove()
        for art in list(self.ax.texts):
            art.remove()

        # Response at integer no. of time constants
        t_tau = self.tau * np.arange(1, 10)
        s_tau = self._calc_response(t_tau)

        self.ax.plot(self.t(), self.s(), '-', color='C0')
        self.ax.plot(t_tau, s_tau, 'o', color='C0')
        self.ax.axhline(y=self.s_0, color='gray', linestyle='-')
        self.ax.axhline(y=self.s_inf, color='gray', linestyle='-')
        self.ax.grid(True)

        for k in range(len(t_tau)):
            self.ax.text(t_tau[k], s_tau[k],
                         rf'   {k+1}$\tau$   ',
                         color='C0', clip_on=True, va='bottom', ha='right')

        ds = abs(self.s_inf - self.s_0)
        s_min = min(self.s_inf, self.s_0) - 0.1*ds
        s_max = max(self.s_inf, self.s_0) + 0.1*ds
        self.ax.set_ylim(s_min, s_max)

        return 0

    def interact(self, tau=None, s_0=None, s_inf=None):
        """Set values and call plotting function."""
        if tau is not None:
            self.tau = tau
        if s_0 is not None:
            self.s_0 = s_0
        if s_inf is not None:
            self.s_inf = s_inf

        self.display()
        return

    def _create_widgets(self):
        """Create widgets for interactive operation."""
        title_widget = ipywidgets.Label(
            'First Order System Frequency Response',
            style=dict(font_weight='bold'))

        slider_layout = {'continuous_update': True,
                         'layout': ipywidgets.Layout(width='50%'),
                         'style': {'description_width': '20%'}}

        text_layout = {'continuous_update': True,
                       'layout': ipywidgets.Layout(width='70%'),
                       'style': {'description_width': '50%'}}

        s_0_widget = ipywidgets.FloatText(
            min=-10, max=10, value=0, step=0.5,
            description='Initial value ',
            readout_format='.1f',
            **text_layout)

        s_inf_widget = ipywidgets.FloatText(
            min=-10, max=10, value=1, step=0.5,
            description='Final value ',
            readout_format='.1f',
            **text_layout)

        tau_widget = ipywidgets.FloatSlider(
            min=0, max=10, value=3, step=0.1,
            description='Time constant [s]',
            readout_format='.1f',
            **slider_layout)

        widget_layout = ipywidgets.VBox([s_0_widget, s_inf_widget])
        widget_layout = ipywidgets.HBox([widget_layout, tau_widget])

        widget_layout = ipywidgets.VBox([title_widget, widget_layout])

        # Export as dictionary
        widget = {'tau': tau_widget,
                  's_0': s_0_widget,
                  's_inf': s_inf_widget}

        w = WidgetLayout(widget_layout, widget)

        return w
